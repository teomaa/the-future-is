# Makefile for adjective_model_larger256_4 inference
# Auto-generated - compiles the inference executable

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Source files
SOURCES = adjective_model_larger256_4_weights.cpp \
          adjective_model_larger256_4.cpp \
          adjective_model_larger256_4_inference.cpp

# Header files
HEADERS = adjective_model_larger256_4.h

OBJECTS = $(SOURCES:.cpp=.o)
TARGET = adjective_model_larger256_4_inference

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean

