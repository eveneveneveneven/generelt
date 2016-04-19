CXX=g++
LDLIBS=-ltesseract -llept `pkg-config --cflags --libs opencv`
TARGET=finding_images_working                        \
	   GetComponentImagesExample            \
	   ResultIteratorExample                \
	   OrientationAndScriptDetectionExample \
	   IteratorExample

RM=rm -f

all:$(TARGET)

clean: 
	$(RM) $(TARGET)

