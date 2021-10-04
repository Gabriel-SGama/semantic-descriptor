# Name of the project
PROJ_NAME=slam.o
PROJ_NAME_DEBUG=debugSlam.o

# .cpp files
CPP_SOURCE=$(wildcard ./src/*.cpp)

# .hpp files
H_SOURCE=$(wildcard ./src/*.hpp)

# Object files
OBJ=$(subst .cpp,.o,$(subst src,objects,$(CPP_SOURCE)))

# Compiler and linker
CC=g++

# Flags for compiler
CC_FLAGS= -c	\
		  -Wall \
		  -msse4 \

CC_FLAGS_DEBUG = -g \
				 -Wall \
		   		 -c \
				 -msse4

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    CC_FLAGS +=-Wall -g
endif

# Command used at clean target
RM = rm -rf

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

#
# Compilation and linking
#
all: objFolder $(PROJ_NAME)

$(PROJ_NAME): $(OBJ)
	@ echo 'Building binary using GCC linker: $@'
	$(CC) $^ -o $@ $(LIBS)
	@ echo 'Finished building binary: $@'
	@ echo ' '

./objects/%.o: ./src/%.cpp ./src/include/%.hpp
	@ echo 'Building target using GCC compiler: $<'
	$(CC) $< $(CC_FLAGS) -o $@ $(LIBS)
	@ echo ' '

./objects/orb_my.o: ./src/orb_my.cpp $(H_SOURCE)
	@ echo 'Building target using GCC compiler: $<'
	$(CC) $< $(CC_FLAGS) -o $@ $(LIBS)
	@ echo ' '

objFolder:
	@ mkdir -p objects

clean:
	@ $(RM) ./objects/*.o $(PROJ_NAME) *~
	@ rmdir objects

.PHONY: all clean
