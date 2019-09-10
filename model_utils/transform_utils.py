from torchvision import transforms

#this file is just a reference for coding
#if you forget how to write a transforms you can check it here

resnet_transforms = {
            'train': transforms.Compose([
                # transforms.Scale(256),
                transforms.RandomSizedCrop(448),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                # transforms.RandomSizedCrop(448),
                transforms.Scale(600),
                transforms.TenCrop(448),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
            ])
}
