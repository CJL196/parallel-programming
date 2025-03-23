# 定义子目录列表
SUBDIRS := $(wildcard */)

# 默认目标
all: run

# 递归执行 make run
run:
	@for dir in $(SUBDIRS); do \
		echo "Running 'make run' in $$dir"; \
		$(MAKE) -C $$dir run || exit 1; \
	done

# 递归执行 make clean
clean:
	@for dir in $(SUBDIRS); do \
		echo "Running 'make clean' in $$dir"; \
		$(MAKE) -C $$dir clean || exit 1; \
	done

# 声明伪目标
.PHONY: all run clean