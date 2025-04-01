from Bio import SeqIO

# 输入FASTA文件路径
input_file = "example_data/output2.fasta"

# 每个新文件包含的最大序列数
batch_size = 20000

# 初始化计数器和文件名
counter = 0
output_file = None

# 遍历输入FASTA文件中的每个序列
for seq_record in SeqIO.parse(input_file, "fasta"):
    # 如果计数器等于0，或者已达到batch_size，请创建一个新的输出FASTA文件
    if counter == 0 or counter % batch_size == 0:
        output_file = f"output_{counter // batch_size + 1}.fasta"

    # 将序列记录写入当前的输出FASTA文件
    with open(output_file, "a") as out_f:
        SeqIO.write(seq_record, out_f, "fasta")

    # 增加计数器
    counter += 1
