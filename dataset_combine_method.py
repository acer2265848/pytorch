
# coding: utf-8

# In[ ]:


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
    def __init__ (self,is_whole,is_cross,method,method2,cross_number_for_test=1,batchsize=24,numworkers=2):
        random_seed = 1
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
# '''
# train data set
# '''
            normal   = pd.read_json("normal_train"+whole+".json")
            abnormal = pd.read_json("abnormal_train"+whole+".json")
            x=normal.append(abnormal)
            x=x.sample(frac=1,random_state=random_seed)

            def pil_loader(path):
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('L')
# '''
# cross=F train
# '''
            lst_method = list()
            lst_method_all = list()
            count=0
            y=list()

            for i in x.batch.unique():
                df=x[x.batch==i]
                lst_method = list()
                lst_2nd_layer = list()
                if np.all(df.status=="abnormal"):
                    path="2_wafer_img"+whole+"/"+method+"/train/abnormal/"
                    path2="2_wafer_img"+whole+"/"+method2+"/train/abnormal/"
                else:
                    path="2_wafer_img"+whole+"/"+method+"/train/normal/"
                    path2="2_wafer_img"+whole+"/"+method2+"/train/normal/"

                for j in x.sensor.unique():
                    img = pil_loader(path+str(i)+"_"+str(j)+".png")
                    img2 = pil_loader(path2+str(i)+"_"+str(j)+".png")
                    a_np = (np.array(img)/255)
                    a2_np = (np.array(img2)/255)
                    a_new = np.zeros((128,128))
                    a_new[np.triu_indices(128, 1)] = a_np[np.triu_indices(127)]
                    a_new[np.tril_indices(128,-1)] = a2_np[np.tril_indices(127)]
                    lst_method.append(a_new)
                arr = np.array(lst_method)


                imgs_comb = np.vstack(i for i in arr)  


                lst_2nd_layer.append(np.array(imgs_comb))
                lst_method_all.append(np.array(lst_2nd_layer))
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1)

            train_x = torch.from_numpy(np.array(lst_method_all)).float()
            train_y = torch.LongTensor(np.array(y))
            train_data = Dataset(train_x,train_y)
            train_loader = DataLoader(dataset=train_data,
                                      batch_size=24,
                                      shuffle=True,
                                      num_workers=2)
            self.train_x=train_x
            self.train_y=train_y
            self.train_loader=train_loader

# '''
# test data set
# '''
            normal   = pd.read_json("normal_validation"+whole+".json")
            abnormal = pd.read_json("abnormal_validation"+whole+".json")
            x=normal.append(abnormal)
            x=x.sample(frac=1,random_state=random_seed)

            # print(x)
            lst_method = list()
            lst_method_all = list()
            count=0
            y=list()
            path=""
            for i in x.batch.unique():
                df=x[x.batch==i]
                lst_method = list()
                lst_2nd_layer = list()
                if np.all(df.status=="abnormal"):
                    path="2_wafer_img"+whole+"/"+method+"/validation/abnormal/"
                    path2="2_wafer_img"+whole+"/"+method2+"/validation/abnormal/"
                else:
                    path="2_wafer_img"+whole+"/"+method+"/validation/normal/"
                    path2="2_wafer_img"+whole+"/"+method2+"/validation/normal/"


                for j in x.sensor.unique():
                    img = pil_loader(path+str(i)+"_"+str(j)+".png")
                    img2 = pil_loader(path2+str(i)+"_"+str(j)+".png")
                    a_np = (np.array(img)/255)
                    a2_np = (np.array(img2)/255)
                    a_new = np.zeros((128,128))
                    a_new[np.triu_indices(128, 1)] = a_np[np.triu_indices(127)]
                    a_new[np.tril_indices(128,-1)] = a2_np[np.tril_indices(127)]
                    lst_method.append(a_new)
                arr = np.array(lst_method)
                
                imgs_comb = np.vstack(i for i in arr) 
                lst_2nd_layer.append(np.array(imgs_comb))
                lst_method_all.append(np.array(lst_2nd_layer))
                
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1) 
            #     if count==1:
            #         break
            test_x = Variable(torch.from_numpy(np.array(lst_method_all)).float()).cuda()
            test_y = torch.LongTensor(np.array(y)).cuda()
            self.test_x=test_x
            self.test_y=test_y        

            print("done")
            
        else:
            import pandas as pd
            import numpy as np
            from PIL import Image
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
                pd.read_json("normal"+whole+"_cross_"+cross[0]+".json"),
                pd.read_json("normal"+whole+"_cross_"+cross[1]+".json"),
                pd.read_json("normal"+whole+"_cross_"+cross[2]+".json"),
                pd.read_json("normal"+whole+"_cross_"+cross[3]+".json"),
                pd.read_json("abnormal"+whole+"_cross_"+cross[0]+".json"),
                pd.read_json("abnormal"+whole+"_cross_"+cross[1]+".json"),
                pd.read_json("abnormal"+whole+"_cross_"+cross[2]+".json"),
                pd.read_json("abnormal"+whole+"_cross_"+cross[3]+".json")
            ]
            x = pd.concat(x)
            x =  x.sample(frac=1,random_state=random_seed)


            def pil_loader(path):
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('L')
            lst_method = list()
            lst_method_all = list()
            count=0
            y=list()

            for i in x.batch.unique():
                df=x[x.batch==i]
                lst_method = list()
                lst_2nd_layer = list()
                if np.all(df.status=="abnormal"):
                    path="2_wafer_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/abnormal/"
                    path2="2_wafer_img_cross"+whole+"/"+method2+"/"+str(df.iloc[0].cross)+"/abnormal/"
                else:
                    path="2_wafer_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/normal/"
                    path2="2_wafer_img_cross"+whole+"/"+method2+"/"+str(df.iloc[0].cross)+"/normal/"
                for j in x.sensor.unique():
                    img = pil_loader(path+str(i)+"_"+str(j)+".png")
                    img2 = pil_loader(path2+str(i)+"_"+str(j)+".png")
                    a_np = (np.array(img)/255)
                    a2_np = (np.array(img2)/255)
                    a_new = np.zeros((128,128))
                    a_new[np.triu_indices(128, 1)] = a_np[np.triu_indices(127)]
                    a_new[np.tril_indices(128,-1)] = a2_np[np.tril_indices(127)]
                    lst_method.append(a_new)
                arr = np.array(lst_method)
                
                imgs_comb = np.vstack(i for i in arr) 
                lst_2nd_layer.append(np.array(imgs_comb))
                lst_method_all.append(np.array(lst_2nd_layer))
                
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1)

            train_x = torch.from_numpy(np.array(lst_method_all)).float()
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
            normal   = pd.read_json("normal"+whole+"_cross_"+str(cross_number_for_test)+".json")
            abnormal = pd.read_json("abnormal"+whole+"_cross_"+str(cross_number_for_test)+".json")
            x=normal.append(abnormal)
            x=x.sample(frac=1)

            # print(x)
            lst_method = list()
            lst_method_all = list()
            count=0
            y=list()
            path=""
            for i in x.batch.unique():
                df=x[x.batch==i]
                lst_method = list()
                lst_2nd_layer = list()
                if np.all(df.status=="abnormal"):
                    path="2_wafer_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/abnormal/"
                    path2="2_wafer_img_cross"+whole+"/"+method2+"/"+str(df.iloc[0].cross)+"/abnormal/"
                else:
                    path="2_wafer_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/normal/"
                    path2="2_wafer_img_cross"+whole+"/"+method2+"/"+str(df.iloc[0].cross)+"/normal/"

                for j in x.sensor.unique():
                    img = pil_loader(path+str(i)+"_"+str(j)+".png")
                    img2 = pil_loader(path2+str(i)+"_"+str(j)+".png")
                    a_np = (np.array(img)/255)
                    a2_np = (np.array(img2)/255)
                    a_new = np.zeros((128,128))
                    a_new[np.triu_indices(128, 1)] = a_np[np.triu_indices(127)]
                    a_new[np.tril_indices(128,-1)] = a2_np[np.tril_indices(127)]
                    lst_method.append(a_new)
                arr = np.array(lst_method)
                    
                imgs_comb = np.vstack(i for i in arr) 
                lst_2nd_layer.append(np.array(imgs_comb))
                lst_method_all.append(np.array(lst_2nd_layer))
                
                if df.iloc[0].status=="normal":
                    y.append(0)
                else:
                    y.append(1)

            test_x = Variable(torch.from_numpy(np.array(lst_method_all)).float()).cuda()
            test_y = torch.LongTensor(np.array(y)).cuda()
            self.test_x=test_x
            self.test_y=test_y        

            print("done")
            
        
print("1 channel combine setting done")

