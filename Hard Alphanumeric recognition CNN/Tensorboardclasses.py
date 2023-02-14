class TensorBoard_handle2():
    # tensorboard --logdir=runs to run tensorboard from terminal

    def __init__ (self, classes, trainloader):
        self.writer = SummaryWriter('runs/fashion_mnist_experiment')
        self.classes = classes
        self.trainloader = trainloader

    def figure_preds(self, output, labels):
        # plot the images in the batch, along with predicted and true labels
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())
        probs = [F.softmax(el, dim=0)[i].item()
                 for i, el in zip(preds, output)]
        fig = plt.figure(figsize=(12, 48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
            #matplotlib_imshow(images[idx], one_channel=True)
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

    # tensorboard.writer.add_scalar(tag, scalar_value, global step)

    def add_scalar(self, tag, scalar, step):
        self.writer.add_scalar(tag, scalar, step)

    def add_figure(self, tag, output, labels, step):
        figure = self.figure_preds(output=output, labels=labels)
        self.writer.add_figure(tag, figure, global_step=step)
    #tensorboard.add_figure(tag,outputs,labels,step) con (inputs,label)=data in trainloader batched
    # e outputs=model(inputs)

    def add_pr_curve(self, class_index, class_probs, class_preds, global_step=0):
        #class_index è la variabile del ciclo entro il quale si chiama la funzione
        #class_preds, class_probs sono liste. vedere ciclo per la definizione
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

class TensorBoard_handle():
    # tensorboard --logdir=runs to run tensorboard from terminal

    def __init__ (self, model, classes, trainloader):
        self.writer = SummaryWriter('runs/fashion_mnist_experiment_1')
        self.model = model
        self.classes = classes
        self.trainloader = trainloader

    def figure_preds(self, images, labels):
        # plot the images in the batch, along with predicted and true labels
        output = self.model(images)
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())  
        probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
        fig = plt.figure(figsize=(12, 48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
            #matplotlib_imshow(images[idx], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.classes[preds[idx]],
                probs[idx] * 100.0,
                self.classes[labels[idx]]),
                color=("green" if preds[idx] == labels[idx].item() else "red"))
        return fig

    def graph(self):
        trainiter = iter(self.trainloader)
        images, labels = trainiter.next()
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image('some_fashion_mnist_images', img_grid)
        self.writer.add_graph(self.model, images)
        self.writer.close()

    def add_scalar(self, tag, scalar, step):
        self.writer.add_scalar(tag, scalar, step)
        
    def add_figure(self, tag, inputs, labels, step):
        figure = self.figure_preds(images=inputs, labels=labels)
        self.writer.add_figure(tag, figure, global_step = step)
    
    def add_pr_curve(self, class_index, class_probs, class_preds, global_step=0):
        #class_index è la variabile del ciclo entro il quale si chiama la funzione
        #class_preds, class_probs sono liste. vedere ciclo per la definizione
        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)
        tensorboard_preds = test_preds == class_index
        tensorboard_probs = test_probs[:, class_index]
        self.writer.add_pr_curve(self.classes[class_index],
                            tensorboard_preds,
                            tensorboard_probs,
                            global_step=global_step)
        self.writer.close()   


# usage commands in main:
#1:
# tensorboard = TensorBoard_handle(model, classes, trainloader)
# tensorboard.graph()
#2:
tensorboard = TensorBoard_handle2(classes, trainloader)
images, img_grid = tensorboard.make_images_grid()
tensorboard.writer.add_image('some_fashion_mnist_images', img_grid)
tensorboard.writer.add_graph(model, images)
tensorboard.writer.close()

#nei cicli per caricare figure e scalari:
#1:
# tensorboard.add_scalar(tag='training loss', scalar=running_loss / t_lapse, step=t*len(trainloader)+i)
# tensorboard.add_figure(tag='predictions', inputs=inputs, labels=labels, step=t*len(trainloader)+i)
#2:
tensorboard.writer.add_scalar('training loss', running_loss / t_lapse, t*len(trainloader)+i)
tensorboard.add_figure('predictions', outputs, labels, t*len(trainloader)+i)
