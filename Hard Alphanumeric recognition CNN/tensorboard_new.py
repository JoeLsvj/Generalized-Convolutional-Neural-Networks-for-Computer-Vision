import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

class TensorBoard_handle():
    # To enable the debugger in TensorBoard, use the flag: --debugger_port <port_number>
    # tensorboard --logdir=runs to run tensorboard from terminal
    def __init__ (self, classes, trainloader):
        self.writer = SummaryWriter('runs/fashion_mnist_experiment_1')
        self.classes = classes
        self.trainloader = trainloader

    @staticmethod
    def tensorboard_imshow(img, one_channel=False):
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
        _, preds_tensor = torch.max(output, 1)
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


tensorboard = TensorBoard_handle(classes, trainloader)
images, img_grid = tensorboard.make_images_grid()
tensorboard.writer.add_image('some_fashion_mnist_images', img_grid)
tensorboard.writer.add_graph(model, images)
tensorboard.add_n_embedding(dataset=trainset, n=300)

tensorboard.writer.add_scalar('training loss', running_loss / t_lapse, t*len(trainloader)+i)
tensorboard.add_figure('predictions', inputs, outputs, labels, step=t*len(trainloader)+i)