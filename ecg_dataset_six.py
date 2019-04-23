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
    def __init__ (self,is_whole,is_cross,method,cross_number_for_test=1,batchsize=24,numworkers=2):
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
                    return img.convert('L')
            lst = list()
            count=0
            lst2=list()
            y=list()

            for i in x.batch.unique():
                df=x[x.batch==i]
                lst = list()
                if np.all(df.status=="abnormal"):
                    path="ecg_img"+whole+"/"+method+"/train/abnormal/"
                else:
                    path="ecg_img"+whole+"/"+method+"/train/normal/"

                for j in x.sensor.unique():
                    img=pil_loader(path+str(i)+"_"+str(j)+".png")
                    lst.append(np.array(img)/255)
                    
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
                                      batch_size=24,
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
                    path="ecg_img"+whole+"/"+method+"/validation/abnormal/"
                else:
                    path="ecg_img"+whole+"/"+method+"/validation/normal/"

                for j in x.sensor.unique():
                    img=pil_loader(path+str(i)+"_"+str(j)+".png")
                    lst.append(np.array(img)/255)
                    
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
            self.test_x=test_x
            self.test_y=test_y        

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
                    return img.convert('L')
            lst = list()
            count=0
            lst2=list()
            y=list()

            for i in x.batch.unique():
                df=x[x.batch==i]
                lst = list()
                if np.all(df.status=="abnormal"):
                    path="ecg_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/abnormal/"
                else:
                    path="ecg_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/normal/"
                for j in x.sensor.unique():
                    img=pil_loader(path+str(i)+"_"+str(j)+".png")
                    lst.append(np.array(img)/255)

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
                                      batch_size=24,
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
                    path="ecg_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/abnormal/"
                else:
                    path="ecg_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/normal/"

                for j in x.sensor.unique():
                    img=pil_loader(path+str(i)+"_"+str(j)+".png")
                    lst.append(np.array(img)/255)
                    
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
            self.test_x=test_x
            self.test_y=test_y        

            print("done")
            
        
print("Function setting done")
