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
		cur.execute("SELECT * FROM article_is_esg")
		for article_url, is_esg in cur.fetchall():
			label = [0, 0]
			cur.execute("SELECT content FROM articles WHERE url = ?", (article_url, ))
			content, = cur.fetchone()
			content = content[:2500]
			if is_esg == 0:
				label[1] = 1
			elif is_esg == 1:
				label[0] = 1
			dataset.append((content, label))
		return dataset


def _weight():
	with database.connect() as con:
		cur = con.cursor()
		cur.execute("SELECT count(*) FROM article_is_esg")
		num_total_samples, = cur.fetchone()
		cur.execute("SELECT count(*) FROM article_is_esg WHERE is_esg = ?", (1, ))
		yes_esg_samples, = cur.fetchone()
		cur.execute("SELECT count(*) FROM article_is_esg WHERE is_esg = ?", (0, ))
		no_esg_samples, = cur.fetchone()
		yes_weight = num_total_samples / (yes_esg_samples * num_classes)
		no_weight = num_total_samples / (no_esg_samples * num_classes)
		return torch.FloatTensor([yes_weight, no_weight])


class Dataset(torch.utils.data.Dataset):

	def __init__(self, encoding):
		self.input_ids = encoding["input_ids"]
		self.token_type_ids = encoding["token_type_ids"]
		self.attention_mask = encoding["attention_mask"]
		self.labels = encoding["labels"]

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, index):
		return {
			"input_ids": self.input_ids[index],
			"token_type_ids": self.token_type_ids[index],
			"attention_mask": self.attention_mask[index],
			"labels": self.labels[index]
		}


def _dataloader(raw_in, raw_out):
	encoding = tokenizer(raw_in, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
	encoding["labels"] = torch.FloatTensor(raw_out)
	dataset = Dataset(encoding)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
	return dataloader


def _train(epoch, dataloader, loss_function):
	model.train()
	total_loss = 0
	num_batches = len(dataloader)
	for index, encoding in enumerate(dataloader):
		input_ids = encoding["input_ids"].to(device)
		token_type_ids = encoding["token_type_ids"].to(device)
		attention_mask = encoding["attention_mask"].to(device)
		labels = encoding["labels"].to(device)
		prediction = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
		loss = loss_function(prediction, labels)
		loss_item = loss.item()
		total_loss = total_loss + loss_item
		loss.backward()
		optimizer.step()
		for param in model.parameters():
			param.grad = None
		print(f"{index}/{num_batches} {loss_item}")
	avg_loss = total_loss / num_batches
	with open("is_esg/output.txt", "a") as f:
		f.write(f"train epoch{epoch} avg_loss: {avg_loss}\n")
	torch.save(model.state_dict(), f"is_esg/{epoch}.model")


def _eval(epoch, dataloader, loss_function):
	model.eval()
	num_batches = len(dataloader)
	num_dataset = len(dataloader.dataset)
	total_loss = 0
	correct = 0
	with torch.no_grad():
		for index, encoding in enumerate(dataloader):
			input_ids = encoding["input_ids"].to(device)
			token_type_ids = encoding["token_type_ids"].to(device)
			attention_mask = encoding["attention_mask"].to(device)
			labels = encoding["labels"].to(device)
			prediction = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
			loss = loss_function(prediction, labels)
			loss_item = loss.item()
			total_loss = total_loss + loss_item
			correct = correct + (prediction.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()
			print(f"{index}/{num_batches} {loss_item}")
		avg_loss = total_loss / num_batches
		accuracy = correct / num_dataset
		with open("is_esg/output.txt", "a") as f:
			f.write(f"eval epoch{epoch} avg_loss: {avg_loss} accuracy: {accuracy}\n")



def main():
	raw_dataset = _raw_dataset()
	weight = _weight()
	print(weight)
	print(raw_dataset[0])
	train_eval_split_index = 103440
	train_raw_dataset = raw_dataset[:train_eval_split_index]
	eval_raw_dataset = raw_dataset[train_eval_split_index:]

	train_raw_in = [item[0] for item in train_raw_dataset]
	train_raw_out = [item[1] for item in train_raw_dataset]

	eval_raw_in = [item[0] for item in eval_raw_dataset]
	eval_raw_out = [item[1] for item in eval_raw_dataset]

	train_dataloader = _dataloader(train_raw_in, train_raw_out)
	eval_dataloader = _dataloader(eval_raw_in, eval_raw_out)

	loss_function = torch.nn.BCEWithLogitsLoss(weight=weight).to(device)

	for epoch in range(epochs):
		_train(epoch, train_dataloader, loss_function)
		_eval(epoch, eval_dataloader, loss_function)


def predict(sentence):
	model.load_state_dict(torch.load("is_esg/3.model"))
	model.eval()
	encoding = tokenizer(sentence, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
	prediction = model(**encoding.to(device)).logits[0]
	prediction = torch.sigmoid(prediction).tolist()
	return prediction


if __name__ == "__main__":
	main()
