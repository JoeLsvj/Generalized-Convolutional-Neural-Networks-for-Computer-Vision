import torch.nn as nn
import torch
import numpy as np
import json, argparse
import random as rn
import cv2 as cv
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time


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

def archit_log(log_name, H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down, c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, classifier_layers, num_classes, learning_rate, epoch):
    # imposto il formato del .txt
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

def archit_data_handle(argus, save_name):
    # lettura dei parametri di architettura della rete da file txt o dalle flag in base
    # a argus.from_txt (e creazione del file di log contenente i parametri letti dalle flag)
    target_path = os.path.dirname(__file__)
    txt_name = os.path.join(target_path, "LetterRec_architecture.txt")
    log_txt_name = os.path.join(target_path, "model_save/LetterRec_architecture_log_"+str(save_name)+".txt")
    txt_flag = argus.from_txt

    if (txt_flag):
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
        classifier_layers = argus.classifier
        num_classes = argus.num_classes
        learning_rate = float(argus.learning_rate)
        epoch = argus.epoch

    # genero il txt con l'architettura
    archit_log(log_name=log_txt_name, H=H, W=W, min_size=min_size, ch_in_0=ch_in_0, ch_out_0=ch_out_0,
                f_1=f_1, f_2=f_2, c_layers_up=c_layers_up, c_layers_down=c_layers_down, c_size=c_size,
                c_stride=c_stride, c_padd=c_padd, c_num=c_num, p_type=p_type, p_size=p_size, p_stride=p_stride,
                p_padd=p_padd, classifier_layers=classifier_layers, num_classes=num_classes, learning_rate=learning_rate, epoch=epoch)

    return H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down, c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, classifier_layers, num_classes, learning_rate, epoch

class train_report_tool():
    # salva un report del training in un file txt
    def __init__(self, target_path, save_name):
        # inizializzo il file txt
        self.rep_path = os.path.join(target_path, "model_save/LetterRec_report_"+str(save_name)+".txt")
        self.rep = open(self.rep_path, "w")
        self.rep.write("step\tloss\n")
    def report(self, t, t_loss):
        # copio un valore di t e loss
        self.rep.write(str(t)+"\t"+str(t_loss)+"\n")
    def close(self):
        self.rep.close()
        print("Report del training salvato in:", self.rep_path)

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
                print("Raggiunta dimensione minima dell'immagine:  ", H, "x", W)
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


def main(argus):
    # per il salvataggio dei dati successivo
    target_path = os.path.dirname(__file__) # current .py directory path
    try:
        os.mkdir(os.path.join(target_path, "model_save")) # provo a creare una cartella, se c'è già proseguo
    except Exception:
        pass
    save_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) # genero un nome 

    # definisco le classi
    classes = ('A', 'B', 'C', 'D', 'J', 'K')

    # ricavo i parametri di architetture della rete dal txt o dalle flag
    H, W, min_size, ch_in_0, ch_out_0, f_1, f_2, c_layers_up, c_layers_down, c_size, c_stride, c_padd, c_num, p_type, p_size, p_stride, p_padd, classifier_layers, num_classes, learning_rate, epoch = archit_data_handle(argus=argus, save_name=save_name)

    # creo il trainset di training
    trainset_dir = os.path.join(target_path, "letters_dataset/")
    trainset = {'A': [], 'B': [], 'C': [], 'D': [], 'J': [], 'K': []}
    # costruisco il dizionario con le immagini categorizzate per classe
    batch_size=4 # immagini per lettera nel trainset
    for i in classes:
        for j in range(0,batch_size):
            trainset[i].append(img_tool(image=trainset_dir+i+str(j)+".png", to_tensor=True))

    # costruzione della rete
    model = LetterRec(num_classes=num_classes, H=H, W=W, min_size=min_size, ch_in_0=ch_in_0, ch_out_0=ch_out_0, f_1=f_1, f_2=f_2, c_layers_up=c_layers_up, c_layers_down=c_layers_down, c_size=c_size, c_stride=c_stride, c_padd=c_padd, c_num=c_num, p_type=p_type, p_size=p_size, p_stride=p_stride, p_padd=p_padd, classifier_layers=classifier_layers)
    model.eval()
    print("\nRete costruita.\n\nInizio fase di training...")
    
    # loss e optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    reptool = train_report_tool(target_path=target_path, save_name=save_name)

    # training
    t_lapse = 1000 # ogni quanto mostrare la loss
    running_loss = 0.0
    for t in range(epoch):
        optimizer.zero_grad()

        # seleziono un immagine casuale dal trainset
        ran_cl = rn.randint(0, len(classes)-1)
        ran_im = rn.randint(0, batch_size-1)
        test_run = trainset[classes[ran_cl]][ran_im]
        
        outputs = model(test_run)
        true_out = np.array([ran_cl])
        true_out = torch.Tensor(true_out)
        true_out = true_out.long()
        loss = loss_fn(outputs, true_out)
        # NOTA: CrossEntropyLoss richiede come secondo argomento un tensore 1D lungo k (batch size)
        # con type "Long" contenente gli indici delle classi corrispondenti a ciascuno dei k esempi forniti 
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if t%t_lapse == (t_lapse-1):
            print(t,"\t", running_loss / t_lapse) # printo la media della loss negli ultimi t_lapse step
            reptool.report(t, running_loss/t_lapse)
            running_loss = 0.0
    print("\nTraining terminato. Loss finale:", loss.item())
    reptool.close()

    # salvo i parametri learnable del modello nel file LetterRec_state.pth
    save = input("\nSalvare i parametri learnable della rete? (y/[n])\t")
    if save in ['y', 'Y']:
        model_path = os.path.join(target_path, "model_save/LetterRec_state_"+str(save_name)+".pth")
        torch.save(model.state_dict(), model_path)
        print("Rete salvata in:", model_path)


if __name__ == '__main__':

    ap = argparse.ArgumentParser(
        description='LetterRec PyTorch implementation by Video Systems.')

    ap.add_argument("-txt", "--from_txt", action='store_true',
                 help="lettura da file txt LetterRec_architecture.txt")
    ap.add_argument("-hi", "--h_in", type=int, required=False,
                 help="dimensione iniziale operativa (height)", default=40)
    ap.add_argument("-wi", "--w_in", type=int, required=False,
                 help="dimensione iniziale operativa (width)", default=40)
    ap.add_argument("-ms", "--min_size", type=int, required=False,
                 help="dimensione minima dell'output della features extraction", default=3)
    ap.add_argument("-ci", "--ch_in_0", type=int, required=False,
                 help="numero canali in ingresso per la prima Conv2d", default=1)
    ap.add_argument("-co", "--ch_out_0", type=int, required=False,
                 help="numero canali in uscita per la prima Conv2d", default=8)
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
                 help="layers della features extraction che aumenta i canali", default=4)
    ap.add_argument("-cld", "--c_layers_down", type=int, required=False,
                 help="layers della features extraction che scende i canali", default=2)
    ap.add_argument("-pt", "--p_type", type=str, required=False,
                 help="type del pooling: max o avg", default="max")
    ap.add_argument("-clf", "--classifier", type=list, required=False,
                 help="struttura della fully connected: numero neuroni per layers", default=[[20, 1],[10,1]])
    ap.add_argument("-nc", "--num_classes", type=int, required=False,
                 help="numero di classi", default=6)
    ap.add_argument("-ep", "--epoch", type=int, required=False,
                 help="iterazioni dell'apprendimento", default=20000)
    ap.add_argument("-lr", "--learning_rate", type=float, required=False,
                 help="rate di apprendimento", default=float(1e-5))
    argus = ap.parse_args()

    argus.formule_conv = [argus.f_1, argus.f_2]
    argus.layers = [argus.c_layers_up, argus.c_layers_down]
    argus.conv2d = [argus.c_size, argus.c_stride, argus.c_padd, argus.c_num]
    argus.pool2d = [argus.p_type, argus.p_size, argus.p_stride, argus.p_padd]

    main(argus)
