import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import json, argparse
import random as rn
import cv2 as cv
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb
import torch.utils
import matplotlib.pyplot as plt
import os
import time
#import tensorflow as tf
#tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

class Rotation(object):
	# Classe usata nel transforms.Compose per ruotare l'immagine di un angolo casuale
	def __init__(self, degrees, value_range=[-1,1], crop=True):
		# 'value_range': range dei valori dei pixel del tensore (default normalizzato tra -1 e 1)
		self.degrees = degrees*3.141/180
		self.crop = crop
		self.value_range = value_range
		self.dtype = torch.cuda.FloatTensor if device=="cuda:0" else torch.FloatTensor

	def __call__(self, x):
		if self.degrees == 0:
			return x
		x = x - self.value_range[0] # rescale dei valori nell'intervallo [0, ...] per comodità
		theta = torch.rand(size=(1,), dtype=torch.float)*self.degrees*2-self.degrees
		# theta = torch.tensor(theta)
		rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
							[torch.sin(theta), torch.cos(theta), 0]])
		cos = np.abs(rot_mat[0, 0])
		sin = np.abs(rot_mat[0, 1])
		h = x.shape[-1]
		w = x.shape[-2]
		# compute the new bounding dimensions of the image
		nW = int((h * sin) + (w * cos))
		nH = int((h * cos) + (w * sin))
		# fattore di rescale per non eccedere le dimensioni originali h e w
		scale_factor = max(nW/w, nH/h)
		if self.crop == False:
			rot_mat = rot_mat*scale_factor
		rot_mat = rot_mat[None, ...].type(self.dtype).repeat(x.shape[0],1,1)
		# affine_grid e gird_sample lavorano su tensori 4D e 5D
		y = x.unsqueeze(0)
		grid = F.affine_grid(rot_mat, y.size(), align_corners=False).type(self.dtype)
		x = F.grid_sample(y, grid, padding_mode="zeros", align_corners=False)
		x = x.squeeze(0)
		x = x + self.value_range[0] # rescale nel value range originale
		return x
	
	def __repr__(self):
		return self.__class__.__name__ + '(degrees={0})'.format(self.degrees)

class AddGaussianNoise(object):
	# Classe usata nel transforms.Compose per aggiungere rumore gaussiano
	def __init__(self, mean=0., std=1., amount=0., value_range=[-1,1]):
		# 'value_range': range dei valori dei pixel del tensore (default normalizzato tra -1 e 1)
		self.std = std
		self.mean = mean
		self.amount = amount
		self.value_range = value_range
		
	def __call__(self, tensor):
		gauss = torch.randn(tensor.size()) * self.std + self.mean
		return torch.clamp(tensor + self.amount*gauss, min=self.value_range[0], max=self.value_range[1])
	
	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def img_tool(image, W=0, H=0, show=False, to_tensor=False):
	# Funzione per importare, ridimensionare e mostrare un immagine.
	# 'image': ndarray numpy o stringa del path (assoluto o relativo) del file immagine in ingresso
	# 'H', 'W': dimensioni desiderate per il resize (lo spazio residuo viene riempito di nero),
	# se non sono specificati non viene fatto il resize 
	# 'rot_angle': float o int, angolo di rotazione, default=0
	# 'noise_percent': percentuale di rumore gaussiano aggiunto
	# 'show': bool, se è True mostra l'immagine con opencv
	# 'to_tensor': bool, se è True restituisce un torch.Tensor (1,1,H,W) anzichè un np.ndarray
	if type(image)==str:
		img = cv.imread(image, 0)
		if img is None:
			# provo a ricostruire il path completo
			# (utile se la cwd è diversa dalla directory in cui si trova il .py in esecuzione)
			target_path = os.path.dirname(__file__)
			image = os.path.join(target_path, image)
			img = cv.imread(image, 0)
		if img is None:
			raise NameError("wrong image name or path")
	elif type(image)==np.ndarray:
		img = image
	else:
		raise TypeError("image argument must be of type str or np.ndarray")
	if H==0 and W==0:
		H, W = img.shape[:2]
	# resize
	img_shape = img.shape
	rapporto = np.minimum(H/img_shape[0], W/img_shape[1])
	new_img_shape = (int(img_shape[0]*rapporto), int(img_shape[1]*rapporto))
	img = cv.resize(img, (new_img_shape[1], new_img_shape[0]))
	blank = np.zeros((H, W), dtype=np.uint8) # type per leggere la matrice immagine a 255 livelli
	for i in range(0,img.shape[0]):
		for j in range(0,img.shape[1]):
			blank[i,j] = img[i,j]
	img = blank
	# show
	if show:
		cv.imshow("Resized image", img)
		cv.waitKey(0)
		cv.destroyAllWindows() 
	# to tensor
	if to_tensor:
		img = np.expand_dims(img, axis=0)
		img = np.expand_dims(img, axis=0)
		img = torch.Tensor(img)
	return img

def archit_log(log_name, H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down, c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, classifier_layers, num_classes, learning_rate, epoch):
	# Crea un file .txt log con l'architettura della rete
	archit = {
		"h_in": H,
		"w_in": W,
		"min_size": min_size,
		"ch_in_0": ch_in_0,
		"ch_out_0": ch_out_0,
		"formule_conv": [f_1, f_2],
		"layers": [c_layers_up, c_layers_down],
		"conv2d": [c_size, c_stride, c_padd, c_num],
		"pool2d": [p_type, p_size, p_stride, p_padd],
		"classifier": classifier_layers,
		"num_classes": num_classes,
		"learning_rate": learning_rate,
		"epoch": epoch
	}
	# produco il file di log
	with open(log_name, "w") as f:
		f.write(json.dumps(archit, indent=4))

def archit_data_handle(argus, save_name, target_path):
	# Lettura dei parametri di architettura della rete da file txt o dalle flag in base
	# a argus.from_txt e argus.load_model e crea un log con i parametri.
	log_name = os.path.join(target_path, "model_save", save_name, "DynamicCNN_architecture_log.txt")
	
	# lettura da txt
	if argus.from_txt:
		txt_name = os.path.join(target_path, "DynamicCNN_architecture.txt")
		with open(txt_name, "r") as f:
			data = f.read()
		archit = json.loads(data)

		H = archit["h_in"]
		W = archit["h_in"]
		min_size = archit["min_size"]
		ch_in_0 = archit["ch_in_0"]
		ch_out_0 = archit["ch_out_0"]
		f_1 = archit["formule_conv"][0]
		f_2 = archit["formule_conv"][1]  
		c_layers_up = archit["layers"][0]
		c_layers_down = archit["layers"][1]
		c_size = archit["conv2d"][0]
		c_stride = archit["conv2d"][1]
		c_padd = archit["conv2d"][2]
		c_num = archit["conv2d"][3]
		p_type = archit["pool2d"][0]
		p_size = archit["pool2d"][1]
		p_stride = archit["pool2d"][2]
		p_padd = archit["pool2d"][3]
		classifier_layers = archit["classifier"]
		num_classes = archit["num_classes"]
		learning_rate = float(archit["learning_rate"])
		epoch = archit["epoch"]

	# lettura da log se si carica un checkpoint
	elif argus.load_model != None:
		load_log_name = os.path.join(target_path, "model_save/DynamicCNN_architecture_log_"+argus.load_model[0]+".txt")
		with open(load_log_name, "r") as f:
			data = f.read()
		archit = json.loads(data)

		H = archit["h_in"]
		W = archit["h_in"]
		min_size = archit["min_size"]
		ch_in_0 = archit["ch_in_0"]
		ch_out_0 = archit["ch_out_0"]
		f_1 = archit["formule_conv"][0]
		f_2 = archit["formule_conv"][1]  
		c_layers_up = archit["layers"][0]
		c_layers_down = archit["layers"][1]
		c_size = archit["conv2d"][0]
		c_stride = archit["conv2d"][1]
		c_padd = archit["conv2d"][2]
		c_num = archit["conv2d"][3]
		p_type = archit["pool2d"][0]
		p_size = archit["pool2d"][1]
		p_stride = archit["pool2d"][2]
		p_padd = archit["pool2d"][3]
		classifier_layers = archit["classifier"]
		num_classes = archit["num_classes"]
		learning_rate = float(archit["learning_rate"])
		epoch = archit["epoch"]

	# lettura da flag
	else:
		H = argus.h_in
		W = argus.w_in
		min_size = argus.min_size
		ch_in_0 = argus.ch_in_0
		ch_out_0 = argus.ch_out_0
		f_1 = argus.f_1
		f_2 = argus.f_2
		c_layers_up = argus.c_layers_up
		c_layers_down = argus.c_layers_down
		c_size = argus.c_size
		c_stride = argus.c_stride
		c_padd = argus.c_padd
		c_num = argus.c_num
		p_type = argus.p_type
		p_size = argus.p_size
		p_stride = argus.p_stride
		p_padd = argus.p_padd
		classifier_layers = json.loads(argus.classifier)
		num_classes = argus.num_classes
		learning_rate = float(argus.learning_rate)
		epoch = argus.epoch

	# creo il txt di log
	archit_log(log_name=log_name, H=H, W=W, min_size=min_size, ch_in_0=ch_in_0, ch_out_0=ch_out_0,
				f_1=f_1, f_2=f_2, c_layers_up=c_layers_up, c_layers_down=c_layers_down, c_size=c_size,
				c_stride=c_stride, c_padd=c_padd, c_num=c_num, p_type=p_type, p_size=p_size, p_stride=p_stride,
				p_padd=p_padd, classifier_layers=classifier_layers, num_classes=num_classes, learning_rate=learning_rate, epoch=epoch)

	return H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down, c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, classifier_layers, num_classes, learning_rate, epoch

class DynamicCNN(nn.Module):
	# Classe di costruzione della rete
	def __init__(self, H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down, c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, classifier_layers, num_classes=10):
		super(DynamicCNN, self).__init__()

		ch_in = ch_in_0
		ch_out = ch_out_0

		self.stop = False
		# flag per interrompere la costruzione della rete e non scendere sotto min_size

		def size_check(H, W, padd, ksize, stride):
			# funzione che interrompe la costruzione della convolutional prima
			# di ridurre la dimensione dell'immagine oltre min_size
			dil = 1 # dilation
			H_new = int((H + 2*padd - dil*(ksize - 1) - 1)/stride + 1)
			W_new = int((W + 2*padd - dil*(ksize - 1) - 1)/stride + 1)
			if ((H_new < min_size) or (W_new < min_size)):
				self.stop = True
				print("\nRaggiunta dimensione minima dell'immagine:  ", H, "x", W)
				return H, W
			return H_new, W_new

		# formule di definizione del numero di canali delle conv2d
		formula_up = lambda x: eval(f_1)
		formula_down = lambda x: eval(f_2)

		# seleziono il tipo di pooling
		if p_type == "max":
			Pool2d = nn.MaxPool2d(p_size, p_stride, p_padd)
		elif p_type == "avg":
			Pool2d = nn.AvgPool2d(p_size, p_stride, p_padd)
		else:
			raise ValueError("Type di Pool2d non valido, scegliere 'max' (default) o 'avg'")

		# costruzione features extraction
		# parte di conv che aumenta i canali
		self.conv = nn.ModuleList([])
		for i in range(0, c_layers_up):
			H, W = size_check(H, W, c_padd, c_size, c_stride)
			if self.stop:
				break
			self.conv.append(nn.Conv2d(ch_in, ch_out, kernel_size=c_size, stride=c_stride, padding=c_padd))
			for _ in range(1, c_num):
				H, W = size_check(H, W, c_padd, c_size, c_stride)
				if self.stop:
					break
				self.conv.append(nn.Conv2d(ch_out, ch_out, kernel_size=c_size, stride=c_stride, padding=c_padd))
			if self.stop:
				break
			self.conv.append(nn.BatchNorm2d(ch_out))
			self.conv.append(nn.ReLU())
			H, W = size_check(H, W, p_padd, p_size, p_stride)
			if self.stop:
				break
			self.conv.append(Pool2d)
			ch_in = ch_out
			ch_out_up = ch_out
			ch_out = ch_out_0*formula_up(i+1)
		# parte di conv che riduce i canali
		for i in range(0, c_layers_down):
			ch_out = int(ch_out_up*formula_down(-i-1))
			H, W = size_check(H, W, c_padd, c_size, c_stride)
			if self.stop:
				break
			self.conv.append(nn.Conv2d(ch_in, ch_out, kernel_size=c_size, stride=c_stride, padding=c_padd))
			self.conv.append(nn.BatchNorm2d(ch_out))
			self.conv.append(nn.ReLU())
			H, W = size_check(H, W, p_padd, p_size, p_stride)
			if self.stop:
				break
			self.conv.append(Pool2d)
			ch_in = ch_out

		# costruzione fully connected (classifier)
		self.classifier = nn.ModuleList([])
		f_in = ch_out*W*H
		for i in range(0, len(classifier_layers)):
			f_out = classifier_layers[i][0]
			f_layers = classifier_layers[i][1]
			self.classifier.append(nn.Linear(f_in, f_out))
			self.classifier.append(nn.BatchNorm1d(f_out))
			self.classifier.append(nn.ReLU())
			self.classifier.append(nn.Dropout(p=0.8))
			for _ in range(1, f_layers):
				self.classifier.append(nn.Linear(f_out, f_out))
				self.classifier.append(nn.BatchNorm1d(f_out))
				self.classifier.append(nn.ReLU())
				self.classifier.append(nn.Dropout(p=0.8))
			f_in = f_out
		self.classifier.append(nn.Linear(f_out, num_classes))

	def forward(self, x):
		for i in enumerate(self.conv):
			x = self.conv[i[0]](x)
		x = x.reshape(x.size(0), -1)
		for i in enumerate(self.classifier):
			x = self.classifier[i[0]](x)
		out = x
		return out

def model_save(save_type, save_name, directory, mod_dict, opt_dict, t, loss, accuracy, save_flag):
	# Funzione che salva il modello
	# 'save_type': str, 'checkpoint' o 'best_model'
	# 'dir': percorso della cartella in cui salvare i file
	path = os.path.join(directory, "model_save", save_name, "DynamicCNN_"+save_type+".pth")
	if save_flag:
		torch.save({
					't': t,
					'model_state_dict': mod_dict,
					'optimizer_state_dict': opt_dict,
					'loss': loss,
					'accuracy': accuracy
				}, path)
		print(save_type, "salvato in:", path)

class TensorBoard_handle():
	# To enable the debugger in TensorBoard, use the flag: --debugger_port <port_number>
	# tensorboard --logdir=runs to run tensorboard from terminal
	def __init__ (self, classes, trainloader, dir_name):
		self.writer = SummaryWriter('runs/'+dir_name)
		self.classes = classes
		self.trainloader = trainloader

	@staticmethod
	def tensorboard_imshow(img, one_channel=False):
		img = img.cpu()
		if one_channel:
			img = img.mean(dim=0)
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		if one_channel:
			plt.imshow(npimg, cmap="Greys")
		else:
			plt.imshow(np.transpose(npimg, (1, 2, 0)))

	def figure_preds(self, images, output, labels):
		# crea un'immagine da caricare sul tensorboard con add_figure o add_image
		# plot the images in the batch, along with predicted and true labels
		_, preds_tensor = torch.max(output.cpu(), 1)
		preds = np.squeeze(preds_tensor.numpy())
		probs = [F.softmax(el, dim=0)[i].item()
				 for i, el in zip(preds, output)]
		fig = plt.figure(figsize=(12, 48))
		for idx in np.arange(4):
			ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
			self.tensorboard_imshow(images[idx], one_channel=True)
			ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
				self.classes[preds[idx]],
				probs[idx] * 100.0,
				self.classes[labels[idx]]),
				color=("green" if preds[idx] == labels[idx].item() else "red"))
		return fig

	def make_images_grid (self):
		trainiter = iter(self.trainloader)
		images, labels = trainiter.next()
		img_grid = torchvision.utils.make_grid(images)
		return images, img_grid
		# tensorboard.writer.add_image('some_fashion_mnist_images', img_grid)
		# tensorboard.writer.add_graph(model, images)
		# tensorboard.writer.close()

	def add_scalar(self, tag, scalar, step):
		# scalar(float) è il float da caricare ad ogni step(int)
		self.writer.add_scalar(tag, scalar, step)

	def add_figure(self, tag, images, output, labels, step):
		#i labels sono ottenibili anche dal trainloader passato alla classe
		figure = self.figure_preds(images=images, output=output, labels=labels)
		self.writer.add_figure(tag, figure, global_step=step)
		#tensorboard.add_figure(tag,outputs,labels,step) con (inputs,label)=data in trainloader batched 

	def add_n_embedding(self, dataset, n=100):
		# Selects n random datapoints and their corresponding labels from a dataset
		inputz, targetz = dataset[:][0], dataset[:][1]
		assert len(inputz) == len(targetz)
		perm = torch.randperm(len(inputz))
		images, labels = inputz[perm][:n], targetz[perm][:n]
		assert len(images) == len(labels)
		# select random images and their target indices and get the class labels for each image
		class_labels = [self.classes[i] for i in labels]
		features = images.view(-1, 28 * 28)
		self.writer.add_embedding(mat=features, metadata=class_labels, label_img=images.unsqueeze(1))
		self.writer.close()
	
	def add_pr_curve(self, class_index, class_probs, class_preds, global_step=0):
		#class_index è la variabile del ciclo entro il quale si chiama la funzione
		#class_preds, class_probs sono liste. vedere ciclo per la definizione e il funzionamento
		test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
		test_preds = torch.cat(class_preds)
		tensorboard_preds = test_preds == class_index
		tensorboard_probs = test_probs[:, class_index]
		self.writer.add_pr_curve(self.classes[class_index],
								 tensorboard_preds,
								 tensorboard_probs,
								 global_step=global_step)
		self.writer.close()

	#tensorboard.writer.close()

class CharactersDataset(object):
	def __init__(self, root_dir, split_rate):
		# 'root_dir' (string): Directory with all the images.
		# creo una lista contenente una serie di lista formate da path immagine e label
		self.classes = sorted(os.listdir(root_dir))
		self.labels = [i for i in range(len(self.classes))]
		self.converter = dict(zip(self.classes, self.labels))
		self.imglist = []
		self.recurrences = {}
		self.imgdict = {}
		for folder in self.classes:
			count = 0
			self.imgdict[folder] = []
			for file in os.listdir(os.path.join(root_dir, folder)):
				if file.endswith(".png") and 'real' in file:
					self.imglist.append([os.path.join(root_dir, folder, file), self.converter[folder]])
					self.imgdict[folder].append(os.path.join(root_dir, folder, file))
					count += 1
			self.recurrences[folder] = count
		# rn.shuffle(self.imglist)
		self.train = []
		self.valid = []
		for i in self.classes:
			x = min(max(int(len(self.imgdict[i])*split_rate), 1), len(self.imgdict[i])-1)
			for j in self.imgdict[i][:x]:
				self.train.append([j, self.converter[i]])
			for j in self.imgdict[i][x:]:
				self.valid.append([j, self.converter[i]])

		# self.valid = self.imglist[int(split_rate*len(self.imglist)):]
		# self.train = self.imglist[:int(split_rate*len(self.imglist))]

class CharactersLoader(torch.utils.data.Dataset):
	def __init__(self, dataset=None, transform=None):
		# 'transform' (callable, optional): Optional transform to be applied on a sample.
		self.imglist = dataset
		self.transform = transform

	def __len__(self):
		return len(self.imglist)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name, label = self.imglist[idx]
		image = np.float32(img_tool(img_name, W=50, H=50))
		sample = [image, np.float32(label)]
		if self.transform:
			sample[0] = self.transform(sample[0])

		return sample

def get_accuracy(dataloader, accname, model):
	# Funzione che ritorna l'accuracy della rete sul testset
	correct = 0
	total = 0
	with torch.no_grad():
		for data in dataloader:
			images, labels = data
			images = images.to(device)
			labels = torch.FloatTensor(labels)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print(accname+'accuracy: %d %%' % (100 * correct / total))
	return (100 * correct / total)

def main(argus):
	# per il salvataggio dei dati successivo
	target_path = os.path.dirname(__file__) # current .py directory path
	save_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) # genero un nome unico
	try:
		os.mkdir(os.path.join(target_path, "model_save"))
	except Exception:
		pass
	try:
		os.mkdir(os.path.join(target_path, "model_save", save_name))
	except Exception:
		pass
	
	# cartella del dataset
	root_dir = os.path.join(target_path, 'Dataset_rotated_chars')

	global device
	device = torch.device("cpu")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# transforms
	transform = transforms.Compose([
		transforms.ToTensor(), 
		transforms.Normalize((0.,), (255.0,)),
		AddGaussianNoise(mean=0., std=1., amount=argus.noise),
		Rotation(degrees=argus.max_rotation),
		# transforms.Resize(size, interpolation=2)
		])

	# datasets
	dataset = CharactersDataset(root_dir=root_dir, split_rate=0.9)
	trainset = CharactersLoader(dataset=dataset.train, transform=transform)
	testset = CharactersLoader(dataset=dataset.valid, transform=transform)
	print('\nImmagini nel trainset:', len(trainset))
	print('Immagini nel testset:', len(testset))
	classes = dataset.classes

	# dataloaders
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=argus.batch_size,
											shuffle=True, num_workers=2, drop_last=True)

	testloader = torch.utils.data.DataLoader(testset, batch_size=argus.batch_size,
											shuffle=False, num_workers=2, drop_last=True)

	# ricavo i parametri di architetture della rete dal txt o dalle flag
	H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down, c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, classifier_layers, num_classes, learning_rate, epoch = archit_data_handle(argus=argus, save_name=save_name, target_path=target_path)

	# costruzione della rete
	model=DynamicCNN(num_classes=num_classes, H=H, W=W, min_size=min_size, ch_in_0=ch_in_0, ch_out_0=ch_out_0, f_1=f_1, f_2=f_2, c_layers_up=c_layers_up, c_layers_down=c_layers_down, c_size=c_size, c_stride=c_stride, c_padd=c_padd, c_num=c_num, p_type=p_type, p_size=p_size, p_stride=p_stride, p_padd=p_padd, classifier_layers=classifier_layers).to(device)

	# creo tensore 1D con i pesi delle classi in base al numero di immagini per label
	recurrences = torch.Tensor(list(dataset.recurrences.values())).to(device)
	class_weights = sum(recurrences)/recurrences

	# loss e optimizer
	loss_fn = nn.CrossEntropyLoss(weight=class_weights)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	# tensorboard initialization
	if argus.tensorboard_flag:
		tensorboard = TensorBoard_handle(classes, trainloader, save_name)
		images, img_grid = tensorboard.make_images_grid()
		tensorboard.writer.add_image('some_fashion_mnist_images', img_grid)
		tensorboard.writer.add_graph(model, images.to(device))
		tensorboard.add_n_embedding(dataset=trainset, n=300) # VA MODIFICATO PER IL NUOVO TRAINSET

	t = 0 # inizializzo passo di apprendimento
	best_accuracy = 1 / len(classes) # inizializzo accuracy (pure chance)

	# carico lo stato del modello dal checkpoint, se richiesto
	if argus.load_model != None:
		checkpoint_path = os.path.join(target_path, "model_save", argus.load_model[0], "/DynamicCNN_"+argus.load_model[1]+".pth")
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		t = checkpoint['t']
		loss = checkpoint['loss']
		best_accuracy = checkpoint['accuracy']
		print("\nCheckpoint",argus.load_model[0] ,"loaded.")
		model.train()
		print("Training resumed.")

	# training
	print("\nInizio training...")
	t_lapse = 20 # ogni quanto mostrare la loss
	while t < epoch:
		running_loss = 0.0
		model.train()
		start = time.time()
		for i, data in enumerate(trainloader, 1):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			inputs = inputs.to(device)
			labels = torch.FloatTensor(labels)
			labels = labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(inputs)
			loss = loss_fn(outputs, labels.type(torch.long))
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			# ogni t_lapse mini-batch printo la running loss e calcolo l'accuracy 
			if i % t_lapse == 0: 
				print('epoch:', t+1, '\tstep:', i, '\tloss:', running_loss / t_lapse)
				#scrive sul tensorboard 
				if argus.tensorboard_flag:
					tensorboard.writer.add_scalar('training loss', running_loss / t_lapse, t*len(trainloader)+i)
					tensorboard.add_figure('predictions', inputs, outputs, labels, step=t*len(trainloader)+i)
				running_loss = 0.0

		end = time.time()
		print('Tempo di calcolo epoca: %.2f'%(end-start), 'seconds')
		
		model.eval()
		_ = get_accuracy(dataloader=trainloader, accname='train_', model=model)
		current_accuracy = get_accuracy(dataloader=testloader, accname='valid_', model=model)
		#scrive sul tensorboard l'accuracy
		if argus.tensorboard_flag:
			tensorboard.writer.add_scalar('accuracy', current_accuracy, t)

		# salvo il modello con la miglior accuracy
		soglia_best = 0.01 # minimo miglioramento dell'accuracy per salvare la rete
		if soglia_best < (current_accuracy/100 - best_accuracy):
			best_accuracy = current_accuracy/100
			model_save(save_type='best_model', save_name=save_name, directory=target_path, mod_dict=model.state_dict(), opt_dict=optimizer.state_dict(), t=t, loss=loss.item(), accuracy=best_accuracy, save_flag=argus.save_checkpoint_best)

		# salvo un checkpoint al termine di ogni epoch
		model_save(save_type='checkpoint', save_name=save_name, directory=target_path, mod_dict=model.state_dict(), opt_dict=optimizer.state_dict(), t=t, loss=loss.item(), accuracy=0.0, save_flag=argus.save_checkpoint_best)
	
		t += 1 # incremento passo di apprendimento

	# caricamento delle precision-recall curves sul tensorboard a fine apprendimento
	# implementare eventualmente nel ciclo di controllo durante l'apprendimento stesso
	if argus.tensorboard_flag:
		print("\ncaricamento sul tensorboard delle precision-recall-curves")
		class_probs = []
		class_preds = []
		with torch.no_grad():
			for data in testloader:
				inputs, labels = data
				inputs = inputs.to(device)
				labels = labels.to(device)
				outputs = model(inputs)
				class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
				_, class_preds_batch = torch.max(outputs, 1)
				class_probs.append(class_probs_batch)
				class_preds.append(class_preds_batch)
		for i in range(len(classes)):
			tensorboard.add_pr_curve(i, class_probs, class_preds)

	print("\nTraining terminato.")

if __name__ == '__main__':

	ap = argparse.ArgumentParser(
		description='DynamicCNN PyTorch implementation by Video Systems.')

	ap.add_argument("-lm", "--load_model", type=list, required=False,
				 help="identificativo del salvataggio da caricare per continuare il training (es. '2020-10-27_13-59-49') e tipo di salvataggio ('checkpoint' o 'best_model')",
				 default=None)
	ap.add_argument("-scb", "--save_checkpoint_best", action='store_false',
				 help="salvataggio automatico del best model e checkpoint durante il training")
	ap.add_argument("-txt", "--from_txt", action='store_true',
				 help="lettura da file txt architecture.txt")
	ap.add_argument("-tbf", "--tensorboard_flag", action='store_true',
				 help="caricamento immagini sul tensorboard")                 
	ap.add_argument("-hi", "--h_in", type=int, required=False,
				 help="dimensione iniziale operativa (height)", default=50)
	ap.add_argument("-wi", "--w_in", type=int, required=False,
				 help="dimensione iniziale operativa (width)", default=50)
	ap.add_argument("-bs", "--batch_size", type=int, required=False,
				 help="dimensione della batch nel trainloader e testloader", default=32)
	ap.add_argument("-mr", "--max_rotation", type=int, required=False,
				 help="massimo angolo di rotazione casuale per l'augmentation del dataset", default=10)
	ap.add_argument("-n", "--noise", type=float, required=False,
				 help="quantità di rumore da aggiungere alle immagini per augmentation", default=0.)                    
	ap.add_argument("-ms", "--min_size", type=int, required=False,
				 help="dimensione minima dell'output della features extraction", default=6)
	ap.add_argument("-ci", "--ch_in_0", type=int, required=False,
				 help="numero canali in ingresso per la prima Conv2d", default=1)
	ap.add_argument("-co", "--ch_out_0", type=int, required=False,
				 help="numero canali in uscita per la prima Conv2d", default=64)
	ap.add_argument("-f1", "--f_1", type=str, required=False,
				 help="formula iterativa di incremento canali della Conv2d", default="2**x")
	ap.add_argument("-f2", "--f_2", type=str, required=False,
				 help="formula iterativa di discesa canali della Conv2d", default="2**(-x)")
	ap.add_argument("-cs", "--c_size", type=int, required=False,
				 help="dimensione kernel della features extraction", default=3)
	ap.add_argument("-cst", "--c_stride", type=int, required=False,
				 help="stride della features extraction", default=1)
	ap.add_argument("-cp", "--c_padd", type=int, required=False,
				 help="padding della features extraction", default=1)
	ap.add_argument("-cn", "--c_num", type=int, required=False,
				 help="numero di strati di features extraction per ciascun blocco", default=1)
	ap.add_argument("-ps", "--p_size", type=int, required=False,
				 help="dimensione kernel del pooling", default=3)
	ap.add_argument("-pst", "--p_stride", type=int, required=False,
				 help="stride del pooling", default=2)
	ap.add_argument("-pp", "--p_padd", type=int, required=False,
				 help="padding del pooling", default=1)
	ap.add_argument("-clu", "--c_layers_up", type=int, required=False,
				 help="layers della features extraction che aumenta i canali", default=5)
	ap.add_argument("-cld", "--c_layers_down", type=int, required=False,
				 help="layers della features extraction che scende i canali", default=5)
	ap.add_argument("-pt", "--p_type", type=str, required=False,
				 help="type del pooling: max o avg", default="max")
	ap.add_argument("-clf", "--classifier", type=str, required=False,
				 help="struttura della fully connected: numero neuroni per layers", default="[[2048,2]]")
	ap.add_argument("-nc", "--num_classes", type=int, required=False,
				 help="numero di classi", default=32)
	ap.add_argument("-ep", "--epoch", type=int, required=False,
				 help="iterazioni dell'apprendimento", default=15)
	ap.add_argument("-lr", "--learning_rate", type=float, required=False,
				 help="rate di apprendimento", default=float(1e-3))
	argus = ap.parse_args()

	argus.formule_conv = [argus.f_1, argus.f_2]
	argus.layers = [argus.c_layers_up, argus.c_layers_down]
	argus.conv2d = [argus.c_size, argus.c_stride, argus.c_padd, argus.c_num]
	argus.pool2d = [argus.p_type, argus.p_size, argus.p_stride, argus.p_padd]

	main(argus)
