import difflib
from fugashi import Tagger
import re
import os

import unicodedata

import torch
torch.backends.quantized.engine = 'qnnpack'
from transformers import BertJapaneseTokenizer
import pytorch_lightning as pl
import requests

def download_file_from_dropbox(url, local_path):
    if not os.path.exists(local_path):
        print(f"{local_path} does not exist. Downloading from Dropbox...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"File downloaded successfully to {local_path}")
        else:
            print(f"Failed to download file from {url}. Status code: {response.status_code}")
    else:
        print(f"{local_path} already exists. Skipping download.")

# Dropboxの共有リンクとローカルのパスのマッピング
files_to_download = [
    {
        "url": "https://www.dropbox.com/scl/fi/z1219662co6y3nmlniw24/model.pth?rlkey=ipniad8nmj2m3dohq922ldy3p&st=8hi06a07&dl=1",
        "local_path": "models/quantize_kouseiyarou_20240721/model.pth"
    },
    {
        "url": "https://www.dropbox.com/scl/fi/pwwub8a9md4q0of5756pj/special_tokens_map.json?rlkey=splxk20i6emjrts62kksdzqo0&st=bmuq0cdd&dl=1",
        "local_path": "models/quantize_kouseiyarou_20240721/special_tokens_map.json"
    },
    {
        "url": "https://www.dropbox.com/scl/fi/y02qoqv6aqf4aaj5qg3kx/tokenizer_config.json?rlkey=e7o4ejs1mrbjex22jsyklbk9y&st=4buiu0a0&dl=1",
        "local_path": "models/quantize_kouseiyarou_20240721/tokenizer_config.json"
    },
    {
        "url": "https://www.dropbox.com/scl/fi/zqlpx0vvgp4qup1edf58r/vocab.txt?rlkey=zfnzl8sjqj6civgdpiyozvqlu&st=f920oxxw&dl=1",
        "local_path": "models/quantize_kouseiyarou_20240721/vocab.txt"
    }
]

os.makedirs('models',exist_ok=True)
os.makedirs('models/quantize_kouseiyarou_20240721',exist_ok=True)

# ファイルをダウンロード
for file_info in files_to_download:
    download_file_from_dropbox(file_info["url"], file_info["local_path"])

class BertForMaskedLM_pl(pl.LightningModule):   
    def __init__(self):
        super().__init__()

class Analyzer():
    def __init__(self):
        self.color_dic = {
            'fix':'<span style="background-color:blue; color: white;">',
            'attention':'<span style="background-color:yellow; color: red;">',
            'end':'</span>'
        }
        self.sentence_end_marks = ('。','」','】','!','?','！','？',)
        self.tokenizer_dir = 'models/quantize_kouseiyarou_20240721'
        self.best_model_path = 'models/quantize_kouseiyarou_20240721/model.pth'
        self.tokenizer = SC_tokenizer.from_pretrained(self.tokenizer_dir)

        model = torch.load(self.best_model_path)
        self.bert_mlm = model.bert_mlm
        del model
        self.tagger = Tagger('-Owakati')

    def get_diff_hl(self, origin, correct):
        """
        文字列の差異をハイライト表示する
        """
        d = difflib.Differ()

        diffs = d.compare(origin, correct)

        result = ''
        diff_texts = []
        for diff in diffs:
            status, _, character = list(diff)
            diff_text = self.decode_char(character)
            if status == '-':
                result += self.color_dic['attention'] + diff_text + self.color_dic['end']
                diff_texts.append(diff_text)
            elif status == '+':
                result += self.color_dic['fix'] + diff_text + self.color_dic['end']
            else:
                result += diff_text

        return result, diff_texts

    def typo_text(self, input_text):
        input_text = self.encode_char(input_text)
        word_nodes = self.tagger(input_text)

        texts = []
        sentence = ""
        for wi, word_node in enumerate(word_nodes):
            sentence += word_node.surface
            if word_node.surface in self.sentence_end_marks:
                texts.append(sentence)
                sentence = ""
            elif wi == len(word_nodes)-1:
                # 最後に。がついていない場合はつけて追加
                texts.append(sentence+'。')

        all_result = ""
        all_diff_texts = []
        for text in texts:
            text = unicodedata.normalize('NFKC', text)
            predict_text = self.predict_cpu(text)
            result, diff_texts = self.get_diff_hl(text, predict_text)
            all_result += result
            all_diff_texts += diff_texts

        return all_result, all_diff_texts

    def predict_cpu(self, text):
        """
        文章を入力として受け、BERTが予測した文章を出力
        """
        # 符号化
        encoding, spans = self.tokenizer.encode_plus_untagged(
            text, return_tensors='pt'
        )
        encoding = { k: v for k, v in encoding.items() }

        # ラベルの予測値の計算
        with torch.no_grad():
            output = self.bert_mlm(**encoding)
            scores = output.logits
            labels_predicted = scores[0].argmax(-1).cpu().numpy().tolist()

        # ラベル列を文章に変換
        predict_text = self.tokenizer.convert_bert_output_to_text(
            text, labels_predicted, spans
        )

        return predict_text

    def encode_char(self, input_text):
        input_text = re.sub("\n",'@',input_text) #改行コード
        input_text = re.sub(" ","■", input_text) #全角スペース
        input_text = re.sub(" ","★", input_text) #半角スペース
        input_text = re.sub(",","●", input_text) #カンマ
        return input_text

    def decode_char(self, input_text):
        input_text = re.sub('@', "\n",input_text) #改行コード
        input_text = re.sub("■", " ", input_text) #全角スペース
        input_text = re.sub("★", " ", input_text) #半角スペース
        input_text = re.sub("●", "," , input_text) #カンマ
        return input_text

    def sentence_end_text(self, input_text):
        input_text = self.encode_char(input_text)
        word_nodes = self.tagger(input_text)
        words = [word_node.surface for word_node in word_nodes]

        sentences = []
        sentence = []
        for word in words:
            sentence.append(word)
            if word in self.sentence_end_marks:
                sentences.append(sentence)
                sentence = []
        
        result = ""
        texts = []
        for si, sentence in enumerate(sentences):
            if len(sentence)<2:
                for wi, word in enumerate(sentence):
                    result += self.decode_char(word)
                continue
            else:
                pass

            if si == 0:
                if len(sentences[si+1])>1 and sentences[si+1][-2] == sentence[-2]:
                    for wi, word in enumerate(sentence):
                        if wi == len(sentence)-2:
                            result += self.color_dic['attention'] + self.decode_char(word) + self.color_dic['end']
                            texts.append(self.decode_char(word))
                        else:
                            result += self.decode_char(word)
                else:
                    for wi, word in enumerate(sentence):
                        result += self.decode_char(word)
            elif si == len(sentences)-1:
                if len(sentences[si-1])>1 and sentences[si-1][-2] == sentence[-2]:
                    for wi, word in enumerate(sentence):
                        if wi == len(sentence)-2:
                            result += self.color_dic['attention'] + self.decode_char(word) + self.color_dic['end']
                        else:
                            result += self.decode_char(word)
                else:
                    for wi, word in enumerate(sentence):
                        result += self.decode_char(word)
            else:
                if (
                    len(sentences[si-1])>1 and sentences[si-1][-2] == sentence[-2]
                ) or (
                    len(sentences[si+1])>1 and sentences[si+1][-2] == sentence[-2]
                ):
                    for wi, word in enumerate(sentence):
                        if wi == len(sentence)-2:
                            result += self.color_dic['attention'] + self.decode_char(word) + self.color_dic['end']
                        else:
                            result += self.decode_char(word)
                else:
                    for wi, word in enumerate(sentence):
                        result += self.decode_char(word)
        return result, texts

    def near_words(self, input_text):
        input_text = self.encode_char(input_text)
        word_nodes = self.tagger(input_text)

        total_words = []
        sentence_words = []
        for wi, word in enumerate(word_nodes):
            sentence_words.append(word)
            if word.surface in self.sentence_end_marks:
                total_words.append(sentence_words)
                sentence_words = []
        del sentence_words
        result = ""
        texts = []
        if len(total_words) == 1:
            result += self.encode_char(input_text)
            return result, texts

        for si, sentence in enumerate(total_words):
            for wi, word in enumerate(sentence):
                if word.pos.split(',')[0] in ("代名詞","名詞","動詞","形容詞"):
                    pass
                else:
                    result += self.decode_char(word.surface)
                    continue
                text = self.decode_char(word.surface)
                if si == 0 and word.surface in [word_node.surface for word_node in total_words[si+1]]:
                    result += self.color_dic['attention'] + text + self.color_dic['end']
                    texts.append(text)
                elif si == len(total_words)-1 and \
                    word.surface in [word_node.surface for word_node in total_words[si-1]]:
                    result += self.color_dic['attention'] + text + self.color_dic['end']
                    texts.append(text)
                elif (si != 0 and si != len(total_words)-1) and \
                    ((word.surface in [word_node.surface for word_node in total_words[si-1]]) or \
                    (word.surface in [word_node.surface for word_node in total_words[si+1]])):
                    result += self.color_dic['attention'] + text + self.color_dic['end']
                    texts.append(text)
                else:
                    result += text
        return result, texts

    def kosoado(self, input_text):
        input_text = self.encode_char(input_text)
        word_nodes = self.tagger(input_text)

        result = ""
        texts = []
        for word_node in word_nodes:
            text = self.decode_char(word_node.surface)
            if word_node.pos.split(',')[0] == "代名詞":
                result += self.color_dic['attention'] + text + self.color_dic['end']
                texts.append(text)
            else:
                result += text
        return result, texts

    def normalize(self, input_text):
        input_text = self.encode_char(input_text)
        normalize_text = unicodedata.normalize('NFKC', input_text)
        result, diff_texts = self.get_diff_hl(input_text, normalize_text)
        return result, diff_texts

class SC_tokenizer(BertJapaneseTokenizer):

    def encode_plus_tagged(
        self, wrong_text, correct_text, max_length=128
    ):
        """
        ファインチューニング時に使用。
        誤変換を含む文章と正しい文章を入力とし、
        符号化を行いBERTに入力できる形式にする。
        """
        # 誤変換した文章をトークン化し、符号化
        encoding = self(
            wrong_text,
            max_length=max_length,
            padding='max_length',
            truncation=True
        )
        # 正しい文章をトークン化し、符号化
        encoding_correct = self(
            correct_text,
            max_length=max_length,
            padding='max_length',
            truncation=True
        )
        # 正しい文章の符号をラベルとする
        encoding['labels'] = encoding_correct['input_ids']

        return encoding

    def encode_plus_untagged(
        self, text, max_length=None, return_tensors=None
    ):
        """
        文章を符号化し、それぞれのトークンの文章中の位置も特定しておく。
        """
        # 文章のトークン化を行い、
        # それぞれのトークンと文章中の文字列を対応づける。
        tokens = [] # トークンを追加していく。
        tokens_original = [] # トークンに対応する文章中の文字列を追加していく。
        words = self.word_tokenizer.tokenize(text) # MeCabで単語に分割

        for word in words:
            # 単語をサブワードに分割
            tokens_word = self.subword_tokenizer.tokenize(word)
            tokens.extend(tokens_word)
            if tokens_word[0] == '[UNK]': # 未知語への対応
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    token.replace('##','') for token in tokens_word
                ])

        # 各トークンの文章中での位置を調べる。（空白の位置を考慮する）
        position = 0
        spans = [] # トークンの位置を追加していく。
        for token in tokens_original:
            l = len(token)
            while 1:
                if token != text[position:position+l]:
                    position += 1
                else:
                    spans.append([position, position+l])
                    position += l
                    break

        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.convert_tokens_to_ids(tokens)
        encoding = self.prepare_for_model(
            input_ids,
            max_length=max_length,
            padding='max_length' if max_length else False,
            truncation=True if max_length else False
        )
        sequence_length = len(encoding['input_ids'])
        # 特殊トークン[CLS]に対するダミーのspanを追加。
        spans = [[-1, -1]] + spans[:sequence_length-2]
        # 特殊トークン[SEP]、[PAD]に対するダミーのspanを追加。
        spans = spans + [[-1, -1]] * ( sequence_length - len(spans) )

        # 必要に応じてtorch.Tensorにする。
        if return_tensors == 'pt':
            encoding = { k: torch.tensor([v]) for k, v in encoding.items() }

        return encoding, spans

    def convert_bert_output_to_text(self, text, labels, spans):
        """
        推論時に使用。
        文章と、各トークンのラベルの予測値、文章中での位置を入力とする。
        そこから、BERTによって予測された文章に変換。
        """
        assert len(spans) == len(labels)

        # labels, spansから特殊トークンに対応する部分を取り除く
        labels = [label for label, span in zip(labels, spans) if span[0]!=-1]
        spans = [span for span in spans if span[0]!=-1]

        # BERTが予測した文章を作成
        predicted_text = ''
        position = 0
        for label, span in zip(labels, spans):
            start, end = span
            if position != start: # 空白の処理
                predicted_text += text[position:start]
            predicted_token = self.convert_ids_to_tokens(label)
            predicted_token = predicted_token.replace('##', '')
            predicted_token = unicodedata.normalize(
                'NFKC', predicted_token
            )
            predicted_text += predicted_token
            position = end

        return predicted_text

if __name__ == '__main__':
    print("初期化開始")
    analyzer = Analyzer()
    print("処理開始")
    result, _ = analyzer.typo_text('自衛隊が基地がある。吾輩の猫である')
    print(f"結果:{result}")