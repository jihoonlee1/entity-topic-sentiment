import database
import torch
import numpy
import transformers
import random


device = torch.device("cuda:0")
num_labels = 49
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
additional_special_tokens = ["[ENT]", "[TPO]"]
tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
epochs = 5
token_max_length = 512
batch_size = 8


def _dataset():
	with database.connect() as con:
		dataset = []
		cur = con.cursor()
		cur.execute("SELECT DISTINCT id FROM temp_topics")
		for row_id, in cur.fetchall():
			cur.execute("SELECT DISTINCT article_url FROM temp_topics WHERE id = ?", (row_id, ))
			url, = cur.fetchone()
			cur.execute("SELECT content FROM articles WHERE url = ?", (url, ))
			content, = cur.fetchone()
			content = content[:2500]
			cur.execute("SELECT topic_id FROM temp_topics WHERE id = ?", (row_id, ))
			topics = []
			for topic_id, in cur.fetchall():
				topics.append(topic_id)
			dataset.append((content, topics))
		random.shuffle(dataset)
		return dataset


def _weight():
	weight = numpy.zeros(num_labels)
	with database.connect() as con:
		cur = con.cursor()
		cur.execute("SELECT count(DISTINCT id) FROM temp_topics")
		num_total_samples, = cur.fetchone()
		cur.execute("SELECT count(*), topic_id FROM temp_topics GROUP BY topic_id")
		for num_topic_count, topic_id in cur.fetchall():
			weight[topic_id] = num_total_samples / (num_topic_count * num_labels)
		return weight


def _label(topic_ids):
	vec = numpy.zeros(num_labels)
	for topic_id in topic_ids:
		vec[topic_id] = 1
	return vec


class Dataset(torch.utils.data.Dataset):

	def __init__(self, dataset):
		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		x, y = self.dataset[index]
		return (x, _label(y))


def train():
	weight = _weight()
	dataset = _dataset()
	loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight)
	train_eval_split = len(dataset) - int((len(dataset) * 0.1))
	train_dataset = dataset[:train_eval_split]
	eval_dataset = dataset[train_eval_split:]

	train_dataloader = torch.utils.data.DataLoader(Dataset(train_dataset), shuffle=True, batch_size=batch_size)
	eval_dataloader = torch.utils.data.DataLoader(Dataset(eval_dataset), shuffle=True, batch_size=batch_size)

	num_train_batches = len(train_dataloader)
	num_eval_batches = len(eval_dataloader)
	num_eval_datasets = len(eval_dataloader.dataset)

	for epoch in range(epochs):
		train_total_loss = 0
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

		with torch.no_grad():
			eval_total_loss = 0
			num_correct = 0
			for idx, (x, y) in enumerate(eval_dataloader):
				encoding = tokenizer(x, max_length=token_max_length, truncation=True, padding="max_length", return_tensors="pt")
				input_ids = encoding["input_ids"].to(device)
				token_type_ids = encoding["token_type_ids"].to(device)
				attention_mask = encoding["attention_mask"].to(device)
				pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
				loss = loss_fn(pred, y.to(device))
				eval_total_loss = eval_total_loss + loss.item()
				num_correct = num_correct + int((pred.argmax(1) == y.argmax(1).to(device)).sum())
				print(f"{idx}/{num_eval_batches} {loss.item()}")
			eval_avg_loss = eval_total_loss / num_eval_batches
			accuracy = num_correct / num_eval_datasets

		torch.save({
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"train_avg_loss": train_avg_loss,
			"eval_avg_loss": eval_avg_loss,
			"eval_accuracy": accuracy
		}, f"{epoch}.model")


#def test():
#	model.eval()
#	for epoch in range(epochs):
#		cp = torch.load(f"esg/{epoch}.model")
#		model.load_state_dict(cp["model_state_dict"])
#		train_avg_loss = cp["train_avg_loss"]
#		eval_avg_loss = cp["eval_avg_loss"]
#		eval_accuracy = cp["eval_accuracy"]
#		sent0 = "Charli D'Amelio and Emma Watson proved they had legs for days at Milan Fashion Week. The social media star and Harry Potter talent were spotted out on the town for one of Milan’s most notable events — Fashion Week. Both ladies made appearances for Prada’s Spring/Summer 2024 Fashion Show. Watson, 33, and D'Amelio, 19, both had one thing in common as they arrived at Prada events in the city — outfits that highlighted their legs!"
#		sent1 = "Reuters: Apple has been putting lots of effort in order to decrease air pollution from growing lots of trees. Microsoft has been putting lots of air emission from their laptop manufacturing. Microsoft has been charged with child labour in china back in 2013."
#		encoding = tokenizer([sent0, sent1], max_length=token_max_length, truncation=True, padding="max_length", return_tensors="pt")
#		input_ids = encoding["input_ids"].to(device)
#		token_type_ids = encoding["token_type_ids"].to(device)
#		attention_mask = encoding["attention_mask"].to(device)
#		pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits[0]
#		pred = torch.sigmoid(pred).tolist()
#		print(pred)


if __name__ == "__main__":
	train()