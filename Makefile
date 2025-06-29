# Compiler and flags
CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -O2 -Isrc -IEigen -Iinclude

# Folders
SRC_DIR := src
INCLUDE_DIR := Include
BUILD_DIR := Build
BIN := log_reg.exe

# Files
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC))
HEADERS := $(wildcard $(SRC_DIR)/*.hpp) $(wildcard $(INCLUDE_DIR)/*.hpp)

# Default target
all: $(BIN)

# Link object files to produce the binary
$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile .cpp files into Build/*.o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target
clean:
	@if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
	@if exist $(BIN) del $(BIN)

.PHONY: all clean