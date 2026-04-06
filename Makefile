UNAME_S := $(shell uname -s)

CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -ffast-math -DNDEBUG
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

RAYLIB_CFLAGS := $(shell pkg-config --cflags raylib 2>/dev/null)
RAYLIB_LIBS := $(shell pkg-config --libs raylib 2>/dev/null)

COMMON_INCLUDES := -I. -I./third_party $(RAYLIB_CFLAGS)
COMMON_HEADERS := $(wildcard *.h) third_party/raygui.h

ifeq ($(UNAME_S),Darwin)
    LDLIBS += $(RAYLIB_LIBS)
else
    LDLIBS += $(RAYLIB_LIBS)
endif

.PHONY: all clean

all: controller build_features

controller: controller.cpp $(COMMON_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(COMMON_INCLUDES) $< -o $@ $(LDFLAGS) $(LDLIBS)

build_features: build_features.cpp $(wildcard *.h)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I. $< -o $@ $(LDFLAGS)

clean:
	rm -f controller build_features controller.exe build_features.exe controller.html
