
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
    def __init__ (self,pre_path,is_whole,is_cross,method,cross_number_for_test=1,batch_size=12,numworkers=2):
        #pre_path = "wafer_img"
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
            x = []
            status_list=["status_1","status_2","status_3","status_4"]
            for k in status_list:
                status_all  = pd.read_json("LP3_"+k+"_train"+whole+".json")
                x.append(status_all)
            x = pd.concat(x,axis=0,ignore_index=True)
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
                if np.all(df.status==1):
                    path=pre_pat+"LP3"+whole+"/"+method+"/train/status_1/"
                elif np.all(df.status==2):
                    path=pre_pat+"LP3"+whole+"/"+method+"/train/status_2/"
                elif np.all(df.status==3):
                    path=pre_pat+"LP3"+whole+"/"+method+"/train/status_3/"
                elif np.all(df.status==4):
                    path=pre_pat+"LP3"+whole+"/"+method+"/train/status_4/"


                list_im = [
                    
                    path+str(i)+'_1.png', 
                    path+str(i)+'_2.png', 
                    path+str(i)+'_3.png', 
                    path+str(i)+'_4.png', 
                    path+str(i)+'_5.png', 
                    path+str(i)+'_6.png', 
                ]

                imgs = [ pil_loader(i) for i in list_im ]
                for k in range(3):
                    imgs_comb = [np.asarray(i) for i in imgs]
                    imgs_comb = np.vstack( (i[:,:,k] for i in imgs_comb ) )
                    imgs_comb = np.array( imgs_comb)
                    lst.append(np.array(imgs_comb)/255)

                arr = np.array(lst)
                if df.iloc[0].status==1:
                    y.append(0)
                elif df.iloc[0].status==2:
                    y.append(1)
                elif df.iloc[0].status==3:
                    y.append(2)
                elif df.iloc[0].status==4:
                    y.append(3)


                lst2.append(arr)    
            #     count+=1
            #     if count==1:
            #         break
            train_x = torch.from_numpy(np.array(lst2)).float()
            train_y = torch.LongTensor(np.array(y))

            train_data = Dataset(train_x,train_y)
            train_loader = DataLoader(dataset=train_data,
                                      batch_size=12,
                                      shuffle=True,
                                      num_workers=2)
            self.train_x=train_x
            self.train_y=train_y
            self.train_loader=train_loader

            #test data set
            x = []
            status_list=["status_1","status_2","status_3","status_4"]
            for k in status_list:
                status_all  = pd.read_json("LP3_"+k+"_validation"+whole+".json")
                x.append(status_all)
            x = pd.concat(x,axis=0,ignore_index=True)
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
                if np.all(df.status==1):
                    path=pre_pat+"LP3"+whole+"/"+method+"/validation/status_1/"
                elif np.all(df.status==2):
                    path=pre_pat+"LP3"+whole+"/"+method+"/validation/status_2/"
                elif np.all(df.status==3):
                    path=pre_pat+"LP3"+whole+"/"+method+"/validation/status_3/"
                elif np.all(df.status==4):
                    path=pre_pat+"LP3"+whole+"/"+method+"/validation/status_4/"


                list_im = [
                    
                    path+str(i)+'_1.png', 
                    path+str(i)+'_2.png', 
                    path+str(i)+'_3.png', 
                    path+str(i)+'_4.png', 
                    path+str(i)+'_5.png', 
                    path+str(i)+'_6.png', 
                ]
                imgs = [ pil_loader(i) for i in list_im ]
                min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
                for k in range(3):
                    imgs_comb = [np.asarray(i) for i in imgs]
                    imgs_comb = np.vstack( (i[:,:,k] for i in imgs_comb ) )
                    imgs_comb = np.array( imgs_comb)
                    lst.append(np.array(imgs_comb)/255)
                arr = np.array(lst)
                if df.iloc[0].status==1:
                    y.append(0)
                elif df.iloc[0].status==2:
                    y.append(1)
                elif df.iloc[0].status==3:
                    y.append(2)
                elif df.iloc[0].status==4:
                    y.append(3)


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
            import matplotlib.pyplot as plt
            import torch
            from torch.autograd import Variable

            if is_whole==True:
                whole="_all"
            else:
                whole=""
            cross=["1","2","3","4","5"]
            cross.remove(str(cross_number_for_test))

            x = []
            status_list=["status_1","status_2","status_3","status_4"]
            for k in status_list:
                for r in cross:
                    status_all  = pd.read_json("LP3_"+k+whole+"_cross_"+r+".json")
                    x.append(status_all)
            x = pd.concat(x,axis=0,ignore_index=True)

            x =  x.sample(frac=1)


            def RGB_pil_loader(path):
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('RGB')
            def gray_pil_loader(path):
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
                if np.all(df.status==1):
                    path=pre_path+"LP3_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/status_1/"
                elif np.all(df.status==2):
                    path=pre_path+"LP3_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/status_2/"
                elif np.all(df.status==3):
                    path=pre_path+"LP3_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/status_3/"
                elif np.all(df.status==4):
                    path=pre_path+"LP3_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/status_4/"


                list_im = [
                    
                    path+str(i)+'_1.png', 
                    path+str(i)+'_2.png', 
                    path+str(i)+'_3.png', 
                    path+str(i)+'_4.png', 
                    path+str(i)+'_5.png', 
                    path+str(i)+'_6.png', 
                ]
                RGB_imgs = [ RGB_pil_loader(i) for i in list_im ]
                gray_imgs = [ gray_pil_loader(i) for i in list_im ]
                min_shape = sorted( [(np.sum(i.size), i.size ) for i in RGB_imgs])[0][1]
                for k in range(3):
                    imgs_comb = [np.asarray(i) for i in RGB_imgs]
                    imgs_comb = np.vstack( (i[:,:,k] for i in imgs_comb ) )
                    imgs_comb = np.array( imgs_comb)
                    lst.append(np.array(imgs_comb)/255)




                arr = np.array(lst)
                if df.iloc[0].status==1:
                    y.append(0)
                elif df.iloc[0].status==2:
                    y.append(1)
                elif df.iloc[0].status==3:
                    y.append(2)
                elif df.iloc[0].status==4:
                    y.append(3)


                lst2.append(arr)    
            #     count+=1
            #     if count==1:
            #         break
            train_x = torch.from_numpy(np.array(lst2)).float()
            train_y = torch.LongTensor(np.array(y))

            train_data = Dataset(train_x,train_y)
            train_loader = DataLoader(dataset=train_data,
                                      batch_size=12,
                                      shuffle=True,
                                      num_workers=2)
            self.train_x=train_x
            self.train_y=train_y
            self.train_loader=train_loader

            #test data set
            x = []
            status_list=["status_1","status_2","status_3","status_4"]
            for k in status_list:
                status_all  = pd.read_json("LP3_"+k+whole+"_cross_"+str(cross_number_for_test)+".json")
                x.append(status_all)
            x = pd.concat(x,axis=0,ignore_index=True)

            
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
                if np.all(df.status==1):
                    path=pre_path+"LP3_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/status_1/"
                elif np.all(df.status==2):
                    path=pre_path+"LP3_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/status_2/"
                elif np.all(df.status==3):
                    path=pre_path+"LP3_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/status_3/"
                elif np.all(df.status==4):
                    path=pre_path+"LP3_img_cross"+whole+"/"+method+"/"+str(df.iloc[0].cross)+"/status_4/"


                list_im = [
                    
                    path+str(i)+'_1.png', 
                    path+str(i)+'_2.png', 
                    path+str(i)+'_3.png', 
                    path+str(i)+'_4.png', 
                    path+str(i)+'_5.png', 
                    path+str(i)+'_6.png', 
                ]
                RGB_imgs = [ RGB_pil_loader(i) for i in list_im ]
                gray_imgs = [ gray_pil_loader(i) for i in list_im ]
                min_shape = sorted( [(np.sum(i.size), i.size ) for i in RGB_imgs])[0][1]
                for k in range(3):
                    imgs_comb = [np.asarray(i) for i in RGB_imgs]
                    imgs_comb = np.vstack( (i[:,:,k] for i in imgs_comb ) )
                    imgs_comb = np.array( imgs_comb)
                    lst.append(np.array(imgs_comb)/255)




                arr = np.array(lst)
                if df.iloc[0].status==1:
                    y.append(0)
                elif df.iloc[0].status==2:
                    y.append(1)
                elif df.iloc[0].status==3:
                    y.append(2)
                elif df.iloc[0].status==4:
                    y.append(3)

                lst2.append(arr)    
            #     if count==1:
            #         break
            test_x = Variable(torch.from_numpy(np.array(lst2)).float()).cuda()
            test_y = torch.LongTensor(np.array(y)).cuda()
            self.test_x=test_x
            self.test_y=test_y        

            print("done")
            
        
print("LP3 color 1 channel Function setting done")

