import database
import random
import torch
import transformers
import numpy


device = torch.device("cuda:0")
num_labels = 3
batch_size = 4
token_max_length = 512
epochs = 10
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
additional_special_tokens = ["[ENT]", "[TPO]"]
tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)


def _dataset():
	with database.connect() as con:
		dataset = []
		sentiment_count_each = [0, 0, 0]
		cur = con.cursor()
		cur.execute("SELECT DISTINCT article_url FROM article_entity_topic_sentiment WHERE topic_id != ?", (-1, ))
		for url, in cur.fetchall():
			cur.execute("SELECT content FROM articles WHERE url = ?", (url, ))
			content, = cur.fetchone()
			content = content[:2500]
			cur.execute("SELECT DISTINCT entity FROM article_entity_topic_sentiment WHERE article_url = ? AND topic_id != ?", (url, -1))
			for entity, in cur.fetchall():
				cur.execute("SELECT start_pos, end_pos FROM article_entity WHERE article_url = ? AND entity = ?", (url, entity))
				start_pos, end_pos = cur.fetchone()
				cur.execute("SELECT topic_id, sentiment FROM article_entity_topic_sentiment WHERE article_url = ? AND entity = ?", (url, entity))
				for topic_id, sentiment in cur.fetchall():
					cur.execute("SELECT name FROM topics WHERE id = ?", (topic_id, ))
					topic_name, = cur.fetchone()
					if content[start_pos:end_pos] == entity:
						content_replaced = f"[TPO] {topic_name} [TPO] " + content[:start_pos] + f"[ENT] {entity} [ENT]" + content[end_pos:]
						dataset.append((content_replaced, sentiment))
						sentiment_count_each[sentiment] = sentiment_count_each[sentiment] + 1
					else:
						if entity in content:
							content_replaced = f"[TPO] {topic_name} [TPO] " + content
							content_replaced = content_replaced.replace(entity, f"[ENT] {entity} [ENT]", 1)
							dataset.append((content_replaced, sentiment))
							sentiment_count_each[sentiment] = sentiment_count_each[sentiment] + 1
		random.shuffle(dataset)
		return (dataset, len(dataset), sentiment_count_each)


def _weight(num_dataset, sentiment_count_each):
	pos, neg, neu = sentiment_count_each
	weight_pos = num_dataset / (pos * num_labels)
	weight_neg = num_dataset / (neg * num_labels)
	weight_neu = num_dataset / (neu * num_labels)
	return torch.FloatTensor([weight_pos, weight_neg, weight_neu])


def _one_hot(index):
	vector = numpy.zeros(num_labels)
	vector[index] = 1
	return vector


class Dataset(torch.utils.data.Dataset):

	def __init__(self, dataset):
		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		x, y = self.dataset[index]
		return (x, _one_hot(y))


def train():
	dataset, num_dataset, sentiment_count_each = _dataset()
	weight = _weight(num_dataset, sentiment_count_each)
	train_eval_split = num_dataset - int(num_dataset * 0.1)
	loss_fn = torch.nn.CrossEntropyLoss(weight=weight).to(device)
	optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)

	train_dataloader = torch.utils.data.DataLoader(Dataset(dataset[:train_eval_split]), batch_size=batch_size, shuffle=True)
	eval_dataloader = torch.utils.data.DataLoader(Dataset(dataset[train_eval_split:]), batch_size=batch_size, shuffle=True)

	num_train_batches = len(train_dataloader)
	num_eval_batches = len(eval_dataloader)
	num_eval_dataset = len(eval_dataloader.dataset)

	for epoch in range(epochs):
		train_total_loss = 0
		eval_total_loss = 0
		eval_correct = 0

		model.train()
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
			eval_accuracy = eval_correct / num_eval_dataset

			torch.save({
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"train_avg_loss": train_avg_loss,
				"eval_avg_loss": eval_avg_loss,
				"eval_accuracy": eval_accuracy
			}, f"sentiment/{epoch}.model")


if __name__ == "__main__":
	train()
