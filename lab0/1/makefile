# 定义目标脚本
SCRIPT := main.py

# 默认目标
all: run

# 运行 Python 脚本
run:
	@echo "Running $(SCRIPT)..."
	@python $(SCRIPT)

# 清理生成的临时文件
clean:
	@echo "Cleaning up..."
	@rm -f __pycache__/*.pyc *.pyc

# 声明伪目标
.PHONY: all run clean