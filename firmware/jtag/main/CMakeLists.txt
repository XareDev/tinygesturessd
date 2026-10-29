set(srcs "main.cpp")

idf_component_register(
    SRCS "${srcs}"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp-dl esp_driver_usb_serial_jtag esp_timer esp_hw_support esp_system heap
)

# Embed the PTQ-INT8 export without ESP-PPQ test tensors. The linker symbol is
# _binary_gesture_ssdlite_no_test_espdl_start (see main.cpp).
set(MODEL_FILE "${CMAKE_SOURCE_DIR}/models/gesture_ssdlite_no_test.espdl")
target_add_binary_data(${COMPONENT_LIB} "${MODEL_FILE}" BINARY)
