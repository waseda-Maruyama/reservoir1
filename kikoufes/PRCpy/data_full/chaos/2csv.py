import numpy as np

# 1. npyファイルを読み込む
data = np.load('C:\\Users\\maruk\\Desktop\\reservoir1\\kikoufes\\PRCpy\\data_full\\chaos\\mackey_glass_t17.npy')

# 2. CSVファイルとして保存する
# fmt='%.18e' (デフォルト) は浮動小数点、整数なら fmt='%d' などを指定
np.savetxt('output.csv', data, delimiter=',', fmt='%s')

print("CSVへの変換が完了しました！")
