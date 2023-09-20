import torch
import database
import transformers


transformers.logging.set_verbosity_error()
device = torch.device("cuda:0")
additional_special_tokens = ["[ENT]", "[TPO]"]
base_model = "bert-base-uncased"
num_classes = 2
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
		cur.execute("SELECT * FROM article_entity_is_relevant")
		for article_url, entity, is_relevant in cur.fetchall():
			cur.execute("SELECT content FROM articles WHERE url = ?", (article_url, ))
			content, = cur.fetchone()
			content = content[:2000]
			cur.execute("SELECT start_pos, end_pos FROM article_entity WHERE article_url = ? AND entity = ?", (article_url, entity))
			start_pos, end_pos = cur.fetchone()
			if content[start_pos:end_pos] != entity:
				continue
			label = [0, 0]
			if is_relevant == 0:
				label[1] = 1
			elif is_relevant == 1:
				label[0] = 1
			content_replaced = content[:start_pos] + "[ENT] " + entity + " [ENT]" + content[end_pos:]
			dataset.append((content_replaced, label))
		return dataset


def _weight():
	with database.connect() as con:
		cur = con.cursor()
		cur.execute("SELECT count(*) FROM article_entity_is_relevant")
		num_total_samples, = cur.fetchone()
		cur.execute("SELECT count(*) FROM article_entity_is_relevant WHERE is_relevant = ?", (1, ))
		yes_relevant_samples, = cur.fetchone()
		cur.execute("SELECT count(*) FROM article_entity_is_relevant WHERE is_relevant = ?", (0, ))
		no_relevant_samples, = cur.fetchone()
		yes_weight = num_total_samples / (yes_relevant_samples * num_classes)
		no_weight = num_total_samples / (no_relevant_samples * num_classes)
		return torch.FloatTensor([yes_weight, no_weight])


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
	with open("is_relevant/output.txt", "a") as f:
		f.write(f"train epoch{epoch} avg_loss: {avg_loss}\n")
	torch.save(model.state_dict(), f"is_relevant/{epoch}.model")


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
		with open("is_relevant/output.txt", "a") as f:
			f.write(f"eval epoch{epoch} avg_loss: {avg_loss} accuracy: {accuracy}\n")


def test():
	pass


def main():
	raw_dataset = _raw_dataset()
	weight = _weight()
	train_eval_split_index = 180000
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
