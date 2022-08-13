# Tool macros
CC := clang
CCFLAGS := -Wall -Wextra -Werror
CCDBGFLAGS := -g
CCOBJFLAGS := $(CCFLAGS) -c
CCLIBS := -lm

# Path macros
SRC_PATH := src
BIN_PATH := bin
OBJ_PATH := obj

# Source files
SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*, .c*)))

# Object files
OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))

# Compile macros
TARGET_NAME := ocr
DBG_TARGET_NAME := ocr_debug
ifeq ($(OS), Windows_NT)
	TARGET_NAME := $(addsuffix .exe, $(TARGET_NAME))
endif

TARGET := $(BIN_PATH)/$(TARGET_NAME)
TARGET_DBG := $(BIN_PATH)/$(DBG_TARGET_NAME)

# Clean files list
DISTCLEAN_LIST = $(OBJ)

CLEAN_LIST = $(TARGET) \
				$(DISTCLEAN_LIST)

# Default rule:
default: clean all

# Make directory rule:
mkdir: makedir

# Non phony rules
$(TARGET) : $(OBJ)
	$(CC) $(CCFLAGS) -o $@ $(OBJ) $(CCLIBS)

$(TARGET_DBG) : $(OBJ)
	$(CC) $(CCFLAGS) $(CCDBGFLAGS) -o $@ $(OBJ) $(CCLIBS)

$(OBJ_PATH)/%.o : $(SRC_PATH)/%.c*
	$(CC) $(CCOBJFLAGS) -o $@ $<

# Phony rules
.PHONY: run
run:
	@./$(TARGET)

.PHONY: makedir
makedir:
	@mkdir $(BIN_PATH) $(OBJ_PATH) $(DEBUG_PATH)

.PHONY: debug
debug: $(TARGET_DBG)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEANING FILES: $(CLEAN_LIST)
	@rm -rf $(CLEAN_LIST)

.PHONY: distclean
distclean:
	@echo CLEANING FILES: $(DISTCLEAN_LIST)
	@rm -rf $(DISTCLEAN_LIST)
