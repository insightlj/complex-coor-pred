# NAME="basic_II"
# model="basic"

NAME="gate_I"
model="gate"


### 在正是运行程序时使用
logfile="log_${NAME}_train.txt"
nohup \
python AlignCoorConfusion/train.py \
--name=$NAME \
--model=$model \
>> AlignCoorConfusion/nohup/$logfile 2>&1 &

### 在debug时使用
# logfile="log_${NAME}_train.txt"
# python AlignCoorConfusion/train_zero.py \
# --name=$NAME \
# --model=$model \
# --device=$device
