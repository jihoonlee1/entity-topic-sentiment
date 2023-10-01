import database
import random
import json
import numpy
import torch
import transformers


transformers.logging.set_verbosity_error()
num_topics = 49
num_sentiments = 2
num_irrelevant = 1
device = torch.device("cuda")
num_labels = num_topics * num_sentiments + num_irrelevant
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
additional_special_tokens = ["[ENT]"]
tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
batch_size = 4
epochs = 10


def _is_balanced(label_count, threshhold):
	for count in label_count:
		if count != threshhold:
			return False
	return True


def _balance_dataset():
	random.seed(42)
	threshhold = 3500
	label_count = [0 for _ in range(num_labels - 1)]
	with database.connect() as con:
		cur = con.cursor()
		cur.execute("SELECT topic_id, entity_id, sentiment, content FROM sentences")
		all_sentences = cur.fetchall()
		num_all_sentences = len(all_sentences)
		cur.execute("SELECT id, name FROM entities")
		all_entities = cur.fetchall()
		num_all_entities = len(all_entities)
		cur.execute("SELECT id FROM topics")
		all_topics = cur.fetchall()
		num_all_topics = len(all_topics)

		irrelevant_count = 0
		cur.execute("SELECT sentence_id, entity, start_pos, end_pos FROM sentence_entity")
		for sentence_id, entity, start_pos, end_pos in cur.fetchall():
			print(irrelevant_count)
			if irrelevant_count == threshhold:
				break
			cur.execute("SELECT 1 FROM irrelevant_entities WHERE sentence_id = ? AND entity = ?", (sentence_id, entity))
			if cur.fetchone() is not None:
				irrelevant_count = irrelevant_count + 1
				content = []
				cur.execute("SELECT content FROM sentences WHERE id = ?", (sentence_id, ))
				irrelevant_sentence, = cur.fetchone()
				len_content = len(irrelevant_sentence)
				irrelevant_sentence = irrelevant_sentence[:start_pos] + "[ENT] " + entity + " [ENT]" + irrelevant_sentence[end_pos:]
				content.append(irrelevant_sentence)
				random_sentence_indexes = random.sample(range(0, num_all_sentences), 10000)
				for index in random_sentence_indexes:
					_, _, _, random_sent = all_sentences[index]
					if len_content > 2400:
						break
					if entity in random_sent:
						continue
					len_content = len_content + len(random_sent)
					content.append(random_sent)
				random.shuffle(content)
				content = " ".join(content)
				label = [98]
				cur.execute("INSERT INTO train VALUES(?,?)", (content, json.dumps(label)))

		while not _is_balanced(label_count, threshhold):
			can_insert = True
			chosen_entity_id, chosen_entity_name = all_entities[random.randint(0, num_all_entities-1)]
			num_relevant_sentences = random.randint(1, 6)
			topic_indexes = random.sample(range(0, num_all_topics), num_relevant_sentences)
			final_rows = []
			for index in topic_indexes:
				topic_id, = all_topics[index]
				cur.execute("SELECT sentiment, content FROM sentences WHERE entity_id = ? AND topic_id = ?", (chosen_entity_id, topic_id))
				rows = cur.fetchall()
				if not rows:
					continue
				num_rows = len(rows)
				sentiment, sentence = rows[random.randint(0, num_rows-1)]
				label_id = (topic_id * num_sentiments) + sentiment
				if label_count[label_id] == threshhold:
					can_insert = False
					break
				final_rows.append((sentence, label_id))
			if can_insert:
				content = []
				label = []
				len_content = 0
				for idx, (sentence, label_id) in enumerate(final_rows):
					label_count[label_id] = label_count[label_id] + 1
					len_content = len_content + len(sentence)
					label.append(label_id)
					if idx == 0:
						sentence = sentence.replace(chosen_entity_name, f"[ENT] {chosen_entity_name} [ENT]", 1)
					content.append(sentence)
				random_sentence_indexes = random.sample(range(0, num_all_sentences), 10000)
				temp = []
				for index in random_sentence_indexes:
					if len_content > 2400:
						break
					topic_id, entity_id, sentiment, sentence = all_sentences[index]
					if entity_id == chosen_entity_id:
						continue
					if topic_id in temp:
						continue
					len_content = len_content + len(sentence)
					content.append(sentence)
				print(label_count)
				random.shuffle(content)
				content = " ".join(content)
				cur.execute("INSERT INTO train VALUES(?,?)", (content, json.dumps(label)))
		con.commit()




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


def train():
	dataset = _dataset()
	train_eval_split = 100000
	train_dataloader = torch.utils.data.DataLoader(Dataset(dataset[:train_eval_split]), batch_size=batch_size)
	eval_dataloader = torch.utils.data.DataLoader(Dataset(dataset[train_eval_split:]), batch_size=batch_size)

	num_train_batches = len(train_dataloader)
	num_eval_batches = len(eval_dataloader)
	num_eval_dataset = len(eval_dataloader.dataset)

	for epoch in range(epochs):
		train_total_loss = 0
		eval_total_loss = 0
		eval_correct = 0

		model.train()
		for idx, (x, y) in enumerate(train_dataloader):
			encoding = tokenizer(x, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
			input_ids = encoding["input_ids"].to(device)
			token_type_ids = encoding["token_type_ids"].to(device)
			attention_mask = encoding["attention_mask"].to(device)
			prediction = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
			loss = loss_fn(prediction, y.to(device))
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			train_total_loss = train_total_loss + loss.item()
			print(f"{idx}/{num_train_batches} {loss.item()}")
		train_avg_loss = train_total_loss / num_train_batches

		model.eval()
		with torch.no_grad():
			for idx, (x, y) in enumerate(eval_dataloader):
				encoding = tokenizer(x, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
				input_ids = encoding["input_ids"].to(device)
				token_type_ids = encoding["token_type_ids"].to(device)
				attention_mask = encoding["attention_mask"].to(device)
				prediction = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
				loss = loss_fn(prediction, y.to(device))
				eval_correct = eval_correct + (prediction.argmax(1) == y.argmax(1).to(device)).type(torch.float).sum().item()
				eval_total_loss = eval_total_loss + loss.item()
				print(f"{idx}/{num_eval_batches} {loss.item()}")
			eval_avg_loss = eval_total_loss / num_eval_batches
			eval_accuracy = eval_correct / num_eval_dataset

			torch.save({
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"train_avg_loss": train_avg_loss,
				"eval_avg_loss": eval_avg_loss,
				"eval_accuracy": eval_accuracy
			}, f"{epoch}.model")


def test():
	for epoch in range(epochs):
		cp = torch.load(f"{epoch}.model")
		print(f"train_avg_loss: {cp['train_avg_loss']}")
		print(f"eval_avg_loss: {cp['eval_avg_loss']}")
		print(f"eval_accuracy: {cp['eval_accuray']}")
		print("-------------------------")



if __name__ == "__main__":
	_balance_dataset()