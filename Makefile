#!/usr/bin/make -f
# Makefile for veja-bass-cab.lv2 #
# --------------------------------- #

include Makefile.mk

NAME = veja-bass-cab

# --------------------------------------------------------------
# Installation path

INSTALL_PATH = /usr/local/lib/lv2
COMPLETE_INSTALL_PATH = $(DESTDIR)$(INSTALL_PATH)/$(NAME).lv2

# --------------------------------------------------------------
# Default target is to build all plugins

all: build
build: $(NAME)-build

cabsim.wisdom:
	@echo "Generating cabsim.wisdom file, this might take a while..."
	fftwf-wisdom -v -n -x -o $@ \
	rof128 rob128 rof192 rob192 rof256 rob256 rof384 rob384 rof512 rob512 rof640 rob640 rof768 rob768 \
	rof1024 rob1024 rof1280 rob1280 rof1536 rob1536 rof2048 rob2048 rof2176 rob2176 rof2304 rob2304 \
	rof2432 rob2432 rof2560 rob2560 rof3072 rob3072 rof4096 rob4096

# --------------------------------------------------------------
# Build rules

$(NAME)-build: $(NAME).lv2/$(NAME)$(LIB_EXT)

$(NAME).lv2/$(NAME)$(LIB_EXT): $(NAME).c
	$(CC) $^ $(BUILD_C_FLAGS) $(LINK_FLAGS) -lm $(SHARED) -o $@

# --------------------------------------------------------------

clean:
	rm -f $(NAME).lv2/$(NAME)$(LIB_EXT)

# --------------------------------------------------------------

install: build
	install -d $(DESTDIR)$(PREFIX)/lib/lv2/$(NAME).lv2

	install -m 644 $(NAME).lv2/*.so  $(DESTDIR)$(PREFIX)/lib/lv2/$(NAME).lv2/
	install -m 644 $(NAME).lv2/*.ttl $(DESTDIR)$(PREFIX)/lib/lv2/$(NAME).lv2/
	install -m 644 $(NAME).lv2/*.txt $(DESTDIR)$(PREFIX)/lib/lv2/$(NAME).lv2/

# --------------------------------------------------------------

