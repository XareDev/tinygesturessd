#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/usb_serial_jtag.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_task_wdt.h"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

static const char *TAG = "TinyGestureSSD";

// Model geometry. These values must match the training configuration and the
// exported ONNX/ESP-DL graph; see docs/architecture.md and models/README.md.
#define IMG_H                128
#define IMG_W                128
#define IMG_C                3
#define IMG_SIZE             (IMG_H * IMG_W * IMG_C)  // 49,152 INT8 bytes

#define GRID_SIZE            8
#define NUM_ANCHORS_PER_CELL 2
#define NUM_TOTAL_ANCHORS    (GRID_SIZE * GRID_SIZE * NUM_ANCHORS_PER_CELL)  // 128
#define NUM_CLASSES          6                         // background is implicit
#define NUM_CLASSES_BG       (NUM_CLASSES + 1)
#define CONF_THRESH          0.5f                      // matches evaluate_espdl()

static const char *CLASS_NAMES[NUM_CLASSES] = {
    "fist", "palm", "like", "ok", "peace", "point"
};

// The PTQ export in models/ has no ESP-PPQ test tensors. It is the same graph
// as models/kd/gesture_ssdlite.espdl; see firmware/jtag/README.md.
extern const uint8_t model_data_start[] asm("_binary_gesture_ssdlite_no_test_espdl_start");
extern const uint8_t model_data_end[]   asm("_binary_gesture_ssdlite_no_test_espdl_end");

// USB-Serial-JTAG carries the binary image frame and the short text replies.
// UART0 is the console (see sdkconfig.defaults), so boot/error logs do not get
// mixed into the image payload. READY is written explicitly to the USB channel;
// do not rely on ESP_LOGI() being visible there.
extern "C" void app_main(void)
{
    usb_serial_jtag_driver_config_t usb_cfg = {
        .tx_buffer_size = 256,
        .rx_buffer_size = IMG_SIZE * 2
    };
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_cfg));

    // Copy the flash-resident blob into PSRAM before constructing the model.
    // ESP-DL's loader accepts this pointer as rodata and detects that the
    // backing memory is PSRAM; keeping the graph in PSRAM avoids flash-cache
    // traffic during inference on the ESP32-S3.
    size_t model_size = model_data_end - model_data_start;
    uint8_t *model_buf = (uint8_t *)heap_caps_aligned_alloc(32, model_size, MALLOC_CAP_SPIRAM);
    if (!model_buf) {
        ESP_LOGE(TAG, "PSRAM allocation failed for model (%u bytes)", (unsigned)model_size);
        return;
    }
    memcpy(model_buf, model_data_start, model_size);

    dl::Model *model = new dl::Model((const char *)model_buf,
                                     fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                                     0,
                                     dl::MEMORY_MANAGER_GREEDY);
    if (model->get_inputs().empty()) {
        ESP_LOGE(TAG, "model loading failed");
        while (true) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    // Write directly into the model-owned input tensor. ESP-PPQ calibrated its
    // exponent (and zero point) for the [0,1] training range, so firmware does
    // not hard-code a quantization scale.
    dl::TensorBase *input_tensor = model->get_inputs().begin()->second;
    if ((size_t)input_tensor->get_size() != (size_t)IMG_SIZE) {
        ESP_LOGE(TAG, "input size mismatch: model=%d, firmware=%d",
                 input_tensor->get_size(), IMG_SIZE);
        while (true) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }
    int8_t *input_ptr = (int8_t *)input_tensor->get_element_ptr();

    ESP_LOGI(TAG, "model: %u bytes, input exponent %d",
             (unsigned)model_size, input_tensor->get_exponent());
    ESP_LOGI(TAG, "input contract: 128x128 RGB, round(pixel / 255 * 2^%d)",
             -input_tensor->get_exponent());

    // The host waits for this line on the USB channel, not on the UART console.
    static const char ready[] = "READY\n";
    usb_serial_jtag_write_bytes(ready, sizeof(ready) - 1, pdMS_TO_TICKS(1000));

    while (true) {
        uint8_t magic = 0;
        while (magic != 0xAA) {
            usb_serial_jtag_read_bytes(&magic, 1, portMAX_DELAY);
        }

        int64_t t_recv_start = esp_timer_get_time();

        int received = 0;
        while (received < IMG_SIZE) {
            int r = usb_serial_jtag_read_bytes((uint8_t *)input_ptr + received,
                                               IMG_SIZE - received,
                                               pdMS_TO_TICKS(5000));
            if (r <= 0) {
                break;
            }
            received += r;
        }
        int64_t t_recv_done = esp_timer_get_time();

        if (received < IMG_SIZE) {
            ESP_LOGE(TAG, "USB timeout: received %d/%d bytes", received, IMG_SIZE);
            continue;
        }

        esp_task_wdt_reset();
        int64_t t0 = esp_timer_get_time();
        model->run(model->get_inputs());
        int64_t t1 = esp_timer_get_time();

        int64_t inference_us = t1 - t0;
        int64_t transfer_in_us = t_recv_done - t_recv_start;

        // Identify the two heads by element count instead of relying on the
        // map iteration order: cls [128,7] and loc [128,4].
        dl::TensorBase *cls_tensor = nullptr;
        dl::TensorBase *loc_tensor = nullptr;
        for (auto &kv : model->get_outputs()) {
            size_t n_elem = (size_t)kv.second->get_size();
            if (n_elem == (size_t)(NUM_TOTAL_ANCHORS * NUM_CLASSES_BG)) {
                cls_tensor = kv.second;
            } else if (n_elem == (size_t)(NUM_TOTAL_ANCHORS * 4)) {
                loc_tensor = kv.second;
            }
        }
        if (!cls_tensor || !loc_tensor) {
            ESP_LOGE(TAG, "unexpected model outputs (expected 896 and 512 elements)");
            continue;
        }

        const int8_t *cls_data = (const int8_t *)cls_tensor->get_element_ptr();
        const int8_t *loc_data = (const int8_t *)loc_tensor->get_element_ptr();
        (void)loc_data;  // boxes are not needed for single-hand classification

        float cls_scale = powf(2.0f, (float)cls_tensor->get_exponent());

        // Softmax over all seven channels, then take the best foreground class
        // per anchor and the best anchor overall. This mirrors the host-side
        // reference evaluator; there is no NMS because the target assumes one
        // hand per frame.
        float best_score = -1.0f;
        int best_class = -1;
        for (int a = 0; a < NUM_TOTAL_ANCHORS; a++) {
            const int8_t *row = cls_data + a * NUM_CLASSES_BG;
            float logits[NUM_CLASSES_BG];
            float max_logit = -1e30f;
            for (int c = 0; c < NUM_CLASSES_BG; c++) {
                logits[c] = row[c] * cls_scale;
                if (logits[c] > max_logit) {
                    max_logit = logits[c];
                }
            }

            float exp_vals[NUM_CLASSES_BG];
            float sum_exp = 0.0f;
            for (int c = 0; c < NUM_CLASSES_BG; c++) {
                exp_vals[c] = expf(logits[c] - max_logit);
                sum_exp += exp_vals[c];
            }

            for (int c = 1; c < NUM_CLASSES_BG; c++) {
                float prob = exp_vals[c] / sum_exp;
                if (prob > best_score) {
                    best_score = prob;
                    best_class = c - 1;
                }
            }
        }

        char result[160];
        int len;
        if (best_score >= CONF_THRESH && best_class >= 0) {
            len = snprintf(result, sizeof(result),
                           "CLASS:%s SCORE:%.3f INFER_US:%lld XFER_IN_US:%lld\n",
                           CLASS_NAMES[best_class], best_score,
                           (long long)inference_us, (long long)transfer_in_us);
        } else {
            len = snprintf(result, sizeof(result),
                           "CLASS:none SCORE:%.3f INFER_US:%lld XFER_IN_US:%lld\n",
                           best_score, (long long)inference_us, (long long)transfer_in_us);
        }
        usb_serial_jtag_write_bytes(result, len, pdMS_TO_TICKS(100));
    }
}
