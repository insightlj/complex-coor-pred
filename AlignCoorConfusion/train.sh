# NAME="basic_II"
# model="basic"
# device=0

NAME="gate_II"
model="gate"
device=2


### 在正是运行程序时使用
logfile="log_${NAME}_train.txt"
nohup \
python utils/AlignCoorConfusion/train.py \
--name=$NAME \
--model=$model \
--device=$device \
>> utils/AlignCoorConfusion/nohup/$logfile 2>&1 &

### 在debug时使用
# logfile="log_${NAME}_train.txt"
# python utils/AlignCoorConfusion/train.py \
# --name=$NAME \
# --model=$model \
# --device=$device
