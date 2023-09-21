import database
import random
import torch
import transformers

transformers.logging.set_verbosity_error()
device = torch.device("cuda:0")
additional_special_tokens = ["[ENT]", "[TPO]"]
base_model = "bert-base-uncased"
num_classes = 49
model = transformers.BertForSequenceClassification.from_pretrained(base_model, num_labels=num_classes).to(device)
tokenizer = transformers.BertTokenizer.from_pretrained(base_model)
tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
model.resize_token_embeddings(len(tokenizer))
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
epochs = 4


def _raw_dataset():
	with database.connect() as con:
		dataset = []
		cur = con.cursor()
		cur.execute("SELECT DISTINCT article_url FROM article_entity_topic_sentiment")
		for article_url, in cur.fetchall():
			cur.execute("SELECT content FROM articles WHERE url = ?", (article_url, ))
			content, = cur.fetchone()
			content = content[:2000]
			cur.execute("SELECT DISTINCT entity FROM article_entity_topic_sentiment WHERE article_url = ?", (article_url, ))
			for entity, in cur.fetchall():
				cur.execute("SELECT start_pos, end_pos FROM article_entity WHERE article_url = ? AND entity = ?", (article_url, entity))
				start_pos, end_pos = cur.fetchone()
				if content[start_pos:end_pos] != entity:
					continue
				content_replaced = content[:start_pos] + "[ENT] " + entity + " [ENT]" + content[end_pos:]
				label = [0 for _ in range(num_classes)]
				cur.execute("SELECT topic_id FROM article_entity_topic_sentiment WHERE article_url = ? AND entity = ?", (article_url, entity))
				for topic_id, in cur.fetchall():
					label[topic_id] = 1
				dataset.append((content_replaced, label))
		random.shuffle(dataset)
		return dataset


def _weight():
	with database.connect() as con:
		weight = [0 for _ in range(num_classes)]
		num_total_samples = 0
		cur = con.cursor()
		cur.execute("SELECT DISTINCT article_url FROM article_entity_topic_sentiment")
		for article_url, in cur.fetchall():
			cur.execute("SELECT content FROM articles WHERE url = ?", (article_url, ))
			content, = cur.fetchone()
			content = content[:2000]
			cur.execute("SELECT DISTINCT entity FROM article_entity_topic_sentiment WHERE article_url = ?", (article_url, ))
			for entity, in cur.fetchall():
				cur.execute("SELECT start_pos, end_pos FROM article_entity WHERE article_url = ? AND entity = ?", (article_url, entity))
				start_pos, end_pos = cur.fetchone()
				if content[start_pos:end_pos] != entity:
					continue
				num_total_samples = num_total_samples + 1
		cur.execute("SELECT id FROM topics")
		for topic_id, in cur.fetchall():
			cur.execute("SELECT count(*) FROM article_entity_topic_sentiment WHERE topic_id = ?", (topic_id, ))
			topic_sample_count, = cur.fetchone()
			topic_weight = num_total_samples / (topic_sample_count * num_classes)
			weight[topic_id] = topic_weight
		return torch.FloatTensor(weight)


class Dataset(torch.utils.data.Dataset):

	def __init__(self, raw_in, raw_out):
		self.raw_in = raw_in
		self.raw_out = raw_out

	def __len__(self):
		return len(self.raw_in)

	def __getitem__(self, index):
		return {
			"raw_in": self.raw_in[index],
			"raw_out": torch.FloatTensor(self.raw_out[index])
		}


def _train(epoch, dataloader, loss_function):
	model.train()
	total_loss = 0
	num_batches = len(dataloader)
	for index, data in enumerate(dataloader):
		raw_in = data["raw_in"]
		labels = data["raw_out"].to(device)
		encoding = tokenizer(raw_in, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
		prediction = model(**encoding.to(device)).logits
		loss = loss_function(prediction, labels)
		loss_item = loss.item()
		total_loss = total_loss + loss_item
		loss.backward()
		optimizer.step()
		for param in model.parameters():
			param.grad = None
		print(f"{index}/{num_batches} {loss_item}")
	avg_loss = total_loss / num_batches
	with open("topics/output.txt", "a") as f:
		f.write(f"train epoch{epoch} avg_loss: {avg_loss}\n")
	torch.save(model.state_dict(), f"topics/{epoch}.model")


def _eval(epoch, dataloader, loss_function):
	model.eval()
	num_batches = len(dataloader)
	num_dataset = len(dataloader.dataset)
	total_loss = 0
	correct = 0
	with torch.no_grad():
		for index, data in enumerate(dataloader):
			raw_in = data["raw_in"]
			labels = data["raw_out"].to(device)
			encoding = tokenizer(raw_in, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
			prediction = model(**encoding.to(device)).logits
			loss = loss_function(prediction, labels)
			loss_item = loss.item()
			total_loss = total_loss + loss_item
			correct = correct + (prediction.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()
			print(f"{index}/{num_batches} {loss_item}")
		avg_loss = total_loss / num_batches
		accuracy = correct / num_dataset
		with open("topics/output.txt", "a") as f:
			f.write(f"eval epoch{epoch} avg_loss: {avg_loss} accuracy: {accuracy}\n")


def main():
	raw_dataset = _raw_dataset()
	weight = _weight()
	train_eval_split_index = 95000
	train_raw_dataset = raw_dataset[:train_eval_split_index]
	eval_raw_dataset = raw_dataset[train_eval_split_index:]

	train_raw_in = [item[0] for item in train_raw_dataset]
	train_raw_out = [item[1] for item in train_raw_dataset]

	eval_raw_in = [item[0] for item in eval_raw_dataset]
	eval_raw_out = [item[1] for item in eval_raw_dataset]

	train_dataset = Dataset(train_raw_in, train_raw_out)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

	eval_dataset = Dataset(eval_raw_in, eval_raw_out)
	eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=4, shuffle=True)

	loss_function = torch.nn.BCEWithLogitsLoss(weight=weight).to(device)

	for epoch in range(epochs):
		_train(epoch, train_dataloader, loss_function)
		_eval(epoch, eval_dataloader, loss_function)


if __name__ == "__main__":
	main()