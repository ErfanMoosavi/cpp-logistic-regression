# Compiler and flags
CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -O2 -Isrc -IEigen

# Folders
SRC_DIR := src
BUILD_DIR := Build
BIN := log_reg.exe

# Files
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC))
HEADERS := $(wildcard $(SRC_DIR)/*.hpp)

# Default target
all: $(BIN)

# Link object files to final binary
$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files into object files inside build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR) $(BIN)

.PHONY: all clean