from torch.utils.data import Dataset, DataLoader
class Dataset(Dataset):
    def __init__(self,train_x,train_y):
        self.train_x=train_x
        self.train_y=train_y
        self.data_len = len(train_x)

    def __getitem__(self, index):

        train_x=self.train_x[index]
        train_y=self.train_y[index]
        return (train_x, train_y)
    def __len__(self):
        return self.data_len
    
class dataset_dataloader:
    def __init__ (self,pre_path,is_whole,is_cross,method,cross_number_for_test=1,batch_size=24,numworkers=2):
        if is_cross==False:
            import pandas as pd
            import numpy as np
            from PIL import Image
            from pyts.visualization import plot_gasf
            import matplotlib.pyplot as plt
            import torch
            from torch.autograd import Variable

            if is_whole==True:
                whole="_all"
            else:
                whole=""

            normal   = pd.read_json("ecg_normal_train"+whole+".json")
            abnormal = pd.read_json("ecg_abnormal_train"+whole+".json")
            x=normal.append(abnormal)
            x=x.sample(frac=1)

            def pil_loader(path):
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('RGB')
            lst = list()
            count=0
            lst2=list()
            y=list()

            for i in x.batch.unique():
                df=x[x.batch==i]
                lst = list()
                if np.all(df.status=="abnormal"):
                    path=pre_path+"ecg_img"+whole+"/"+method+"/train/abnormal/"
                else:
                    path=pre_path+"ecg_img"+whole+"/"+method+"/train/normal/"

                list_im = [
                    path+str(i)+'_0.png', 
                    path+str(i)+'_1.png', 
                ]
                imgs = [ pil_loader(i) for i in list_im ]
                for k in range(3):
                    imgs_comb = [np.asarray(i) for i in imgs]
                    imgs_comb = np.vstack( (i[:,:,k] for i in imgs_comb ) )
                    imgs_comb = np.array( imgs_comb)
                    lst.append(np.array(imgs_comb)/255)

                arr = np.array(lst)
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1)

                lst2.append(arr)    
            #     count+=1
            #     if count==1:
            #         break
            train_x = torch.from_numpy(np.array(lst2)).float()
            train_y = torch.LongTensor(np.array(y))

            train_data = Dataset(train_x,train_y)
            train_loader = DataLoader(dataset=train_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=2)
            self.train_x=train_x
            self.train_y=train_y
            self.train_loader=train_loader

            #test data set
            normal   = pd.read_json("ecg_normal_validation"+whole+".json")
            abnormal = pd.read_json("ecg_abnormal_validation"+whole+".json")
            x=normal.append(abnormal)
            x=x.sample(frac=1)

            # print(x)
            lst = list()
            count=0
            lst2=list()
            y=list()
            path=""
            for i in x.batch.unique():
                df=x[x.batch==i]
                lst = list()
                if np.all(df.status=="abnormal"):
                    path=pre_path+"ecg_img"+whole+"/"+method+"/validation/abnormal/"
                else:
                    path=pre_path+"ecg_img"+whole+"/"+method+"/validation/normal/"

                list_im = [
                    path+str(i)+'_0.png', 
                    path+str(i)+'_1.png', 

                ]

                imgs = [ pil_loader(i) for i in list_im ]
                min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
                for k in range(3):
                    imgs_comb = [np.asarray(i) for i in imgs]
                    imgs_comb = np.vstack( (i[:,:,k] for i in imgs_comb ) )
                    imgs_comb = np.array( imgs_comb)
                    lst.append(np.array(imgs_comb)/255)
                arr = np.array(lst)
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1)

                lst2.append(arr)    
            #     if count==1:
            #         break
            test_x = Variable(torch.from_numpy(np.array(lst2)).float()).cuda()
            test_y = torch.LongTensor(np.array(y)).cuda()
            test_data = Dataset(test_x,test_y)
            test_loader = DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0)
            self.test_x=test_x
            self.test_y=test_y
            self.test_loader=test_loader  

            print("done")
            
        else:
            import pandas as pd
            import numpy as np
            from PIL import Image
            from pyts.visualization import plot_gasf
            import matplotlib.pyplot as plt
            import torch
            from torch.autograd import Variable

            if is_whole==True:
                whole="_all"
            else:
                whole=""
            cross=["1","2","3","4","5"]
            cross.remove(str(cross_number_for_test))
            x = [  
                pd.read_json("ecg_normal"+whole+"_cross_"+cross[0]+".json"),
                pd.read_json("ecg_normal"+whole+"_cross_"+cross[1]+".json"),
                pd.read_json("ecg_normal"+whole+"_cross_"+cross[2]+".json"),
                pd.read_json("ecg_normal"+whole+"_cross_"+cross[3]+".json"),
                pd.read_json("ecg_abnormal"+whole+"_cross_"+cross[0]+".json"),
                pd.read_json("ecg_abnormal"+whole+"_cross_"+cross[1]+".json"),
                pd.read_json("ecg_abnormal"+whole+"_cross_"+cross[2]+".json"),
                pd.read_json("ecg_abnormal"+whole+"_cross_"+cross[3]+".json")
            ]
            x = pd.concat(x)
            x =  x.sample(frac=1)


            def pil_loader(path):
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('RGB')
            lst = list()
            count=0
            lst2=list()
            y=list()

            for i in x.batch.unique():
                df=x[x.batch==i]
                lst = list()
                if np.all(df.status=="abnormal"):
                    path=pre_path+"ecg_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/abnormal/"
                else:
                    path=pre_path+"ecg_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/normal/"

                list_im = [
                    path+str(i)+'_0.png', 
                    path+str(i)+'_1.png', 
                ]
                imgs = [ pil_loader(i) for i in list_im ]
                min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
                for k in range(3):
                    imgs_comb = [np.asarray(i) for i in imgs]
                    imgs_comb = np.vstack( (i[:,:,k] for i in imgs_comb ) )
                    imgs_comb = np.array( imgs_comb)
                    lst.append(np.array(imgs_comb)/255)


                arr = np.array(lst)
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1)

                lst2.append(arr)    
            #     count+=1
            #     if count==1:
            #         break
            train_x = torch.from_numpy(np.array(lst2)).float()
            train_y = torch.LongTensor(np.array(y))

            train_data = Dataset(train_x,train_y)
            train_loader = DataLoader(dataset=train_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=2)
            self.train_x=train_x
            self.train_y=train_y
            self.train_loader=train_loader

            #test data set
            normal   = pd.read_json("ecg_normal"+whole+"_cross_"+str(cross_number_for_test)+".json")
            abnormal = pd.read_json("ecg_abnormal"+whole+"_cross_"+str(cross_number_for_test)+".json")
            x=normal.append(abnormal)
            x=x.sample(frac=1)

            # print(x)
            lst = list()
            count=0
            lst2=list()
            y=list()
            path=""
            for i in x.batch.unique():
                df=x[x.batch==i]
                lst = list()
                if np.all(df.status=="abnormal"):
                    path=pre_path+"ecg_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/abnormal/"
                else:
                    path=pre_path+"ecg_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/normal/"

                list_im = [
                    path+str(i)+'_0.png', 
                    path+str(i)+'_1.png', 
                ]
                imgs = [ pil_loader(i) for i in list_im ]
                min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
                for k in range(3):
                    imgs_comb = [np.asarray(i) for i in imgs]
                    imgs_comb = np.vstack( (i[:,:,k] for i in imgs_comb ) )
                    imgs_comb = np.array( imgs_comb)
                    lst.append(np.array(imgs_comb)/255)
                arr = np.array(lst)
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1)

                lst2.append(arr)    
            #     if count==1:
            #         break
            test_x = Variable(torch.from_numpy(np.array(lst2)).float()).cuda()
            test_y = torch.LongTensor(np.array(y)).cuda()
            test_data = Dataset(test_x,test_y)
            test_loader = DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0)
            self.test_x=test_x
            self.test_y=test_y
            self.test_loader=test_loader      

            print("done")
            
        
print("ECG color 1 channelFunction setting done")
