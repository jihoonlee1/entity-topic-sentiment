import database
import numpy
import transformers
import torch
import random


device = torch.device("cuda:0")
num_topics = 49
num_sentiments = 3
num_irrelevant = 1
batch_size = 8
token_max_length = 512
epochs = 5
num_labels = (num_topics * num_sentiments) + num_irrelevant
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
additional_special_tokens = ["[ENT]"]
tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)


def _dataset():
	label_count = {}
	for i in range(num_labels):
		label_count[i] = 0
	dataset = []
	with database.connect() as con:
		cur = con.cursor()
		cur.execute("SELECT DISTINCT article_url FROM article_entity_topic_sentiment")
		for url, in cur.fetchall():
			cur.execute("SELECT content FROM articles WHERE url = ?", (url, ))
			content, = cur.fetchone()
			content = content[:2500]
			cur.execute("SELECT DISTINCT entity FROM article_entity_topic_sentiment WHERE article_url = ?", (url, ))
			for entity, in cur.fetchall():
				content_replaced = content
				cur.execute("SELECT start_pos, end_pos FROM article_entity WHERE article_url = ? AND entity = ?", (url, entity))
				start_pos, end_pos = cur.fetchone()
				can_insert = False
				if content[start_pos:end_pos] == entity:
					content_replaced = content[:start_pos] + "[ENT] " + entity + " [ENT]" + content[end_pos:]
					can_insert = True
				else:
					if entity in content:
						content_replaced = content.replace(entity, f"[ENT] {entity} [ENT]")
						can_insert = True
				if not can_insert:
					continue
				labels = []
				cur.execute("SELECT topic_id, sentiment FROM article_entity_topic_sentiment WHERE article_url = ? AND entity = ?", (url, entity))
				topic_sentiment = cur.fetchall()
				check_topic, check_sentiment = topic_sentiment[0]
				if check_topic == -1 and check_sentiment == -1:
					label_count[147] = label_count[147] + 1
					labels.append(147)
				elif check_topic != -1:
					for topic_id, sentiment_id in topic_sentiment:
						label_index = (topic_id * num_sentiments) + sentiment_id
						label_count[label_index] = label_count[label_index] + 1
						labels.append(label_index)
				dataset.append((content_replaced, labels))
		random.shuffle(dataset)
		return (dataset, len(dataset), label_count)


def _label(indexes):
	vec = numpy.zeros(num_labels)
	for index in indexes:
		vec[index] = 1
	return vec


class Dataset(torch.utils.data.Dataset):

	def __init__(self, dataset):
		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		x, y = self.dataset[index]
		return (x, _label(y))


def _weight(num_total_samples, label_count):
	weight = numpy.zeros(num_labels)
	for key, val in label_count.items():
		if val == 0:
			continue
		weight[key] = num_total_samples / (val * num_labels)
	return torch.FloatTensor(weight)


def train():
	dataset, num_total_samples, label_count = _dataset()
	weight = _weight(num_total_samples, label_count)
	loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight).to(device)
	train_eval_split = num_total_samples - int(num_total_samples * 0.1)

	train_dataloader = torch.utils.data.DataLoader(Dataset(dataset[:train_eval_split]), shuffle=True, batch_size=batch_size)
	eval_dataloader = torch.utils.data.DataLoader(Dataset(dataset[train_eval_split:]), shuffle=True, batch_size=batch_size)

	num_train_batches = len(train_dataloader)
	num_eval_batches = len(eval_dataloader)
	num_eval_total_dataset = len(eval_dataloader.dataset)

	for epoch in range(epochs):
		model.train()
		train_total_loss = 0
		eval_total_loss = 0
		eval_correct = 0
		for idx, (x, y) in enumerate(train_dataloader):
			encoding = tokenizer(x, max_length=token_max_length, truncation=True, padding="max_length", return_tensors="pt")
			input_ids = encoding["input_ids"].to(device)
			token_type_ids = encoding["token_type_ids"].to(device)
			attention_mask = encoding["attention_mask"].to(device)
			pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
			loss = loss_fn(pred, y.to(device))
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			train_total_loss = train_total_loss + loss.item()
			print(f"{idx}/{num_train_batches} {loss.item()}")
		train_avg_loss = train_total_loss / num_train_batches

		model.eval()
		with torch.no_grad():
			for idx, (x, y) in enumerate(eval_dataloader):
				encoding = tokenizer(x, max_length=token_max_length, truncation=True, padding="max_length", return_tensors="pt")
				input_ids = encoding["input_ids"].to(device)
				token_type_ids = encoding["token_type_ids"].to(device)
				attention_mask = encoding["attention_mask"].to(device)
				pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
				loss = loss_fn(pred, y.to(device))
				eval_total_loss = eval_total_loss + loss.item()
				eval_correct = eval_correct + (pred.argmax(1) == y.argmax(1).to(device)).type(torch.float).sum().item()
			eval_avg_loss = eval_total_loss / num_eval_batches
			eval_accuracy = eval_correct / num_eval_total_dataset

			torch.save({
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"train_avg_loss": train_avg_loss,
				"eval_avg_loss": eval_avg_loss,
				"eval_accuracy": eval_accuracy
			}, f"{epoch}.model")


if __name__ == "__main__":
	train()
