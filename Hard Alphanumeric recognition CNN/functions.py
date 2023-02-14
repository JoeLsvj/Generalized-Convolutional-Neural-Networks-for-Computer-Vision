
def rotate_bound(image, angle):
    # Ruota l'immagine dell'angolo specificato
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))

def get_rot_mat(theta, shape):
    rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    h = shape[0]
    w = shape[1]
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # fattore di rescale per non eccedere le dimensioni originali h e w
    scale_factor = max(nW/w, nH/h)
    return rot_mat*scale_factor

def rot_img_tensor(x, degrees, value_range=[-1,1], dtype=torch.FloatTensor):
    # Applica una rotazione casuale di angolo in un range [-degrees, degrees] a un tensore
    # 'x': tensore (B, C, H, W), immagine da ruotare
    # 'degrees': angolo massimo di rotazione
    # 'value_range': range dei valori dei pixel del tensore (default normalizzato tra -1 e 1)
    x = x - value_range[0]
    theta = torch.randint(low=-degrees, high=degrees, size=(1,), dtype=torch.float)
    rot_mat = get_rot_mat(theta, (x.shape[-1], x.shape[-2]))[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=False).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    x = x + value_range[0]
    return x

class Rotation(object):
    # Classe usata nel transforms.Compose per ruotare l'immagine di un angolo casuale
    def __init__(self, degrees, value_range=[-1,1]):
        # 'value_range': range dei valori dei pixel del tensore (default normalizzato tra -1 e 1)
        self.degrees = degrees
        self.value_range = value_range
        self.dtype = torch.FloatTensor
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def __call__(self, x):
        x = x - self.value_range[0] # rescale dei valori nell'intervallo [0, ...] per comodità
        theta = torch.randint(low=-self.degrees, high=self.degrees, size=(1,), dtype=torch.float)
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

def add_noise(image, noise_percent=0):
    # Aggiunge rumore gaussiano
    # 'image': ndarray numpy 2D (grayscale)
    # 'noise_percent': percentuale di rumore aggiunto
    row, col = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.1
    gauss = 255*(np.random.normal(mean,sigma,(row,col)))
    gauss = gauss.reshape(row,col)
    noisy = np.uint8(np.minimum(np.maximum(image + (noise_percent/100)*gauss, 0), 255))
    return noisy

def add_noise_tensor(image, amount=0, value_range=[-1, 1], device=torch.device("cpu")):
    # Applica rumore gaussiano a un tensore 3d sul secondo e terzo asse
    # 'value_range': range dei valori dei pixel del tensore (default normalizzato tra -1 e 1)
    var = 0.1**0.5
    gauss = (var*torch.randn(image.shape[1], image.shape[2]))
    image = image + amount*gauss.to(device)
    image = torch.clamp(image, min=value_range[0], max=value_range[1])
    return image

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

def img_tool(image, W=0, H=0, rot_angle=0, noise_percent=0, show=False, to_tensor=False):
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
    # rotate
    if rot_angle!=0:
        img = rotate_bound(img, rot_angle)
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
    img = add_noise(img, noise_percent=noise_percent)
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
    log_name = os.path.join(target_path, "model_save/DynamicCNN_architecture_log_"+str(save_name)+".txt")
    
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

def model_save(save_type, save_name, directory, mod_dict, opt_dict, t, loss, accuracy):
    # Funzione che salva il modello
    # 'save_type': str, 'checkpoint' o 'best_model'
    # 'dir': percorso della cartella in cui salvare i file
    path = os.path.join(directory, "model_save/DynamicCNN_"+save_type+"_"+str(save_name)+".pth")
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
    def __init__ (self, classes, trainloader):
        self.writer = SummaryWriter('runs/fashion_mnist_experiment_1')
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
        inputz, targetz = dataset.data, dataset.targets
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
 