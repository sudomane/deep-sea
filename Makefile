# Tool macros
CC := clang
CCFLAGS :=
DBGFLAGS := -g
CCFLAGS_TEST := $(CCFLAGS) -I/usr/local/include/criterion
CCOBJFLAGS := $(CCFLAGS) -c
CCLIBS := -lm 
CCLIBS_TEST := $(CCLIBS) -lcriterion

# Path macros
SRC_PATH := src
SRC_TEST_PATH := test
BIN_PATH := bin
OBJ_PATH := obj
DEBUG_PATH := debug

# Source files
SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*, .c*)))
SRC_TEST := $(filter-out $(SRC_PATH)/main.c, $(SRC)) $(foreach x, $(SRC_TEST_PATH), $(wildcard $(addprefix $(x)/*, .c*)))

# Object files
OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))
OBJ_DEBUG := $(addprefix $(DEBUG_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))
OBJ_TEST := $(filter-out $(OBJ_PATH)/main.o, $(OBJ)) $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC_TEST))))) 

# Compile macros
TARGET_NAME := ocr
TARGET_NAME_TEST := test
ifeq ($(OS), Windows_NT)
	TARGET_NAME := $(addsuffix .exe, $(TARGET_NAME))
endif

TARGET := $(BIN_PATH)/$(TARGET_NAME)
TARGET_DEBUG := $(DEBUG_PATH)/$(TARGET_NAME)
TARGET_TEST := $(BIN_PATH)/$(TARGET_NAME_TEST)

# Clean files list
DISTCLEAN_LIST = $(OBJ) \
					$(OBJ_DEBUG)

CLEAN_LIST = $(TARGET) \
				$(TARGET_DEBUG) \
				$(TARGET_TEST) \
				$(DISTCLEAN_LIST)

# Default rule:
default: clean all

# Non phony rules
$(TARGET) : $(OBJ)
	$(CC) $(CCFLAGS) -o $@ $(OBJ) $(CCLIBS)

$(OBJ_PATH)/%.o : $(SRC_PATH)/%.c*
	$(CC) $(CCOBJFLAGS) -o $@ $< $(CCLIBS)

$(DBG_PATH)/%.o : $(SRC_PATH)/%.c*
	$(CC) $(CCOBJFLAGS) $(DBGFLAGS) -o $@ $< $(CCLIBS)

$(TARGET_DEBUG) : $(OBJ_DEBUG)
	$(CC) $(CCFLAGS) $(DBGFLAGS) $(OBJ_DEBUG) -o $@ $(CCLIBS)

$(TARGET_TEST) : $(OBJ_TEST)
	@echo $(SRC_TEST)
	@echo $(OBJ_TEST)
	$(CC) $(CCFLAGS_TEST) -o $@ $(OBJ_TEST) $(CCLIBS_TEST) 

# Phony rules
.PHONY: run
run:
	@./$(TARGET)

.PHONY: makedir
makedir:
	@mkdir $(BIN_PATH) $(OBJ_PATH) $(DEBUG_PATH)

.PHONY: all
all: $(TARGET)

.PHONY: debug
debug: $(TARGET_DEBUG)

.PHONY: test
test: $(TARGET_TEST)

.PHONY: clean
clean:
	@echo CLEANING FILES: $(CLEAN_LIST)
	@rm -rf $(CLEAN_LIST)

.PHONY: distclean
distclean:
	@echo CLEANING FILES: $(DISTCLEAN_LIST)
	@rm -rf $(DISTCLEAN_LIST)
