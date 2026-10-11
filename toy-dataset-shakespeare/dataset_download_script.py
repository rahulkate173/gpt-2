from urllib import request 
from rich import print 
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
output_file = "input.txt"
request.urlretrieve(url, output_file)
print("Dataset downloaded !!")