import torch.nn as nn
import torch
import numpy as np
import json, argparse
import random as rn
import cv2 as cv
import torch.optim as optim
import matplotlib.pyplot as plt
import os


def img_tool(image, H=0, W=0, show=False, to_tensor=True):
    # Funzione per importare, ridimensionare e mostrare un immagine.
    # 'image': ndarray numpy o stringa del path (assoluto o relativo) del file immagine in ingresso
    # 'H', 'W': dimensioni desiderate per il resize (lo spazio residuo viene riempito di nero),
    # se non sono specificati non viene fatto il resize 
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
    # resize
    if H!=0 and W!=0:
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

def archit_data_load(save_identif, save_dir):
    # lettura dei parametri di architettura della rete da file txt
    save_log_path = os.path.join(save_dir, "LetterRec_architecture_log_"+save_identif+".txt")
    txt_name = save_log_path
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
    print("Parametri dell'architettura importati:")
    for i in archit:
        print(i,":\t",archit[i])
    print()
    return H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down, c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, classifier_layers, num_classes, learning_rate, epoch

class LetterRec(nn.Module):
    def __init__(self, H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down, c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, classifier_layers, num_classes=10):
        super(LetterRec, self).__init__()

        ch_in = ch_in_0
        ch_out = ch_out_0

        self.stop = False
        # flag per interrompere la costruzione della rete e non scendere sotto min_size

        def size_check(H, W, padd, ksize, stride):
            dil = 1
            H_new = int((H + 2*padd - dil*(ksize - 1) - 1)/stride + 1)
            W_new = int((W + 2*padd - dil*(ksize - 1) - 1)/stride + 1)
            if ((H_new < min_size) or (W_new < min_size)):
                self.stop = True
                print("Raggiunta dimensione minima dell'immagine:  ", H, "x", W, "\n")
                return H, W
            return H_new, W_new

        formula_up = lambda x: eval(f_1)
        formula_down = lambda x: eval(f_2)

        if p_type == "max":
            Pool2d = nn.MaxPool2d(p_size, p_stride, p_padd)
        elif p_type == "avg":
            Pool2d = nn.AvgPool2d(p_size, p_stride, p_padd)
        else:
            raise ValueError("Type di Pool2d non valido, scegliere 'max' (default) o 'avg'")

        # costruzione features extraction
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

# selezionare qui il salvataggio
save_identif = ""
save_dir = os.path.join(os.path.dirname(__file__), "model_save/") # assumo che model_save e questo .py siano nella stessa cartella

save_log_path = os.path.join(save_dir, "LetterRec_architecture_log_"+save_identif+".txt")
save_state_path = os.path.join(save_dir, "LetterRec_state_"+save_identif+".pth")
testset_dir = os.path.join(os.path.dirname(__file__), "letters_dataset/")

# ricavo i parametri di architettura della rete dal txt
H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down, c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, classifier_layers, num_classes, learning_rate, epoch = archit_data_load(save_identif=save_identif, save_dir=save_dir)

# definisco le classi
classes = ('A', 'B', 'C', 'D', 'J', 'K')

# creo il testset
testset = {'A': [], 'B': [], 'C': [], 'D': [], 'J': [], 'K': []}
# costruisco il dizionario con le immagini categorizzate per classe
batch_size=5 # quante immagini per lettera nel testset
for i in classes:
    for j in range(0, batch_size):
        testset[i].append(img_tool(image=testset_dir+i+str(j)+".png", to_tensor=True))

# load model state_dict
model = LetterRec(num_classes=num_classes, H=H, W=W, min_size=min_size, ch_in_0=ch_in_0, ch_out_0=ch_out_0, f_1=f_1, f_2=f_2, c_layers_up=c_layers_up, c_layers_down=c_layers_down, c_size=c_size, c_stride=c_stride, c_padd=c_padd, c_num=c_num, p_type=p_type, p_size=p_size, p_stride=p_stride, p_padd=p_padd, classifier_layers=classifier_layers)
model.load_state_dict(torch.load(save_state_path))
print("Model loaded.\n")
model.eval()

loss_fn = nn.CrossEntropyLoss()
for i, letter in enumerate(testset, 0):
    for j in range(0,batch_size):
        # passo le immagini di testset una ad una
        class_index = i
        image_index = j
        test_run = testset[classes[class_index]][image_index]
        outputs = model(test_run)
        true_out = np.array([class_index])
        true_out = torch.Tensor(true_out)
        true_out = true_out.long()
        loss = loss_fn(outputs, true_out)
        print("Input image:",classes[class_index]+str(image_index)+".png",
            "\tBest guess:\t", classes[outputs.detach().numpy().argmax()],
            "\tLoss:", loss.item())
        selected_image = os.path.join(testset_dir, classes[class_index]+str(image_index)+".png")
        img_tool(image=selected_image, H=200, W=200, show=True, to_tensor=False)
