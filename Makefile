CXX=g++

OPT = -I lib -O2 -mcmodel=medium  -fopenmp -w 

CXXFLAGS = $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: P-Tucker 

P-Tucker: P-Tucker.cpp 
	$(CXX) $(CXXFLAGS)  -o $@  $<

demo: P-Tucker.cpp
	g++ -I lib -o P-Tucker P-Tucker.cpp -O2 -fopenmp -w -mcmodel=medium
	./P-Tucker sample/input.txt sample/result 3 10 20


.PHONY: clean

clean:
	rm -f P-Tucker

