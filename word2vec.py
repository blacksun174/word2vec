from gensim.models import KeyedVectors
import io,sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
model = KeyedVectors.load_word2vec_format("entity_vector/entity_vector.model.bin", binary=True)

results = model.most_similar(positive=[u'[パリ]',u'[日本]'],negative=[u'[東京]'])

print("日本 + パリ - 東京 = ")
for result in results:
    print(result)
