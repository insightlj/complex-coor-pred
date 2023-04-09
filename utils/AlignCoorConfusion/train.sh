NAME="gate_I"
model="gate"
device=7
logfile="log_${NAME}_train.txt"

nohup \
python utils/AlignCoorConfusion/train.py \
--name=$NAME \
--model=$model \
--device=$device \
>> utils/AlignCoorConfusion/nohup/$logfile 2>&1 &

# python utils/AlignCoorConfusion/train.py \
# --name=$NAME \
# --model=$model \
# --device=$device