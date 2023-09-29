import database
import random
import json
import numpy
import torch
import transformers


num_topics = 49
num_sentiments = 2
device = torch.device("cuda")
num_labels = num_topics * num_sentiments
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
additional_special_tokens = ["[ENT]"]
tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)
model.resize_token_embeddings(len(tokenizer))


def _is_balanced(label_count, threshhold):
	for _, val in label_count.items():
		if val != threshhold:
			return False
	return True


def _balance_dataset():
	random.seed(0) # 770498
	threshhold = 3500
	label_count = {}
	dataset = []
	for i in range(num_labels):
		label_count[i] = 0
	with database.connect() as con:
		cur = con.cursor()
		cur.execute("SELECT topic_id, entity_id, sentiment, content FROM sentences")
		all_sentences = cur.fetchall()
		num_all_sentences = len(all_sentences)
		count = 0
		while not _is_balanced(label_count, threshhold):
			print(count)
			count = count + 1
			chosen_sent_idx = random.randint(0, num_all_sentences-1)
			chosen_topic_id, chosen_entity_id, chosen_sentiment, chosen_content = all_sentences[chosen_sent_idx]
			chosen_label_id = (chosen_topic_id * num_sentiments) + chosen_sentiment
			if label_count[chosen_label_id] == threshhold:
				continue
			cur.execute("SELECT topic_id, entity_id, sentiment, content FROM sentences WHERE entity_id = ? AND topic_id != ?", (chosen_entity_id, chosen_topic_id))
			same_entity_sentences = cur.fetchall()
			num_same_entity_sentences = random.randint(0, 5)
			same_entity_sentence_indexes = random.sample(range(0, len(same_entity_sentences)-1), num_same_entity_sentences)
			can_insert = True
			for index in same_entity_sentence_indexes:
				same_entity_topic_id, _, same_entity_sentiment, same_entity_content = same_entity_sentences[index]
				sane_entity_label_id = (same_entity_topic_id * num_sentiments) + same_entity_sentiment
				if label_count[sane_entity_label_id] == threshhold:
					can_insert = False
					break
			if can_insert:
				contents = []
				label = []
				len_content = len(chosen_content)
				label_count[chosen_label_id] = label_count[chosen_label_id] + 1
				label.append(chosen_label_id)
				cur.execute("SELECT name FROM entities WHERE id = ?", (chosen_entity_id, ))
				chosen_entity_name, = cur.fetchone()
				chosen_content_replaced = chosen_content.replace(chosen_entity_name, f"[ENT] {chosen_entity_name} [ENT]", 1)
				contents.append(chosen_content_replaced)
				for index in same_entity_sentence_indexes:
					same_entity_topic_id, _, same_entity_sentiment, same_entity_content = same_entity_sentences[index]
					same_entity_label_id = (same_entity_topic_id * num_sentiments) + same_entity_sentiment
					label_count[same_entity_label_id] = label_count[same_entity_label_id] + 1
					len_content = len_content + len(same_entity_content)
					label.append(same_entity_label_id)
					contents.append(same_entity_content)
				cur.execute("SELECT count(*) FROM sentences WHERE entity_id = ?", (chosen_entity_id, ))
				num_chosen_entity_sentences, = cur.fetchone()
				max_offset = num_all_sentences - num_chosen_entity_sentences
				random_offsets = random.sample(range(0, max_offset), 30)
				for offset in random_offsets:
					if len_content > 2400:
						break
					cur.execute("SELECT content FROM sentences WHERE entity_id != ? LIMIT 1 OFFSET ?", (chosen_entity_id, offset))
					random_content, = cur.fetchone()
					len_content = len_content + len(random_content)
					contents.append(random_content)
				random.shuffle(contents)
				content = " ".join(contents)
				cur.execute("INSERT INTO train VALUES(?,?)", (content, json.dumps(label)))


def _dataset():
	with database.connect() as con:
		dataset = []
		cur = con.cursor()
		cur.execute("SELECT content, label FROM train")
		for content, label in cur.fetchall():
			dataset.append((content, json.loads(label)))
		return dataset


def _label(label_ids):
	vector = numpy.zeros(num_labels)
	for label_id in label_ids:
		vector[label_id] = 1
	return vector


class Dataset(torch.utils.data.Dataset):

	def __init__(self, dataset):
		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		x, y = self.dataset[index]
		return (x, _label(y))

