# def getnet():
    
#    class Net(nn.Module):
#        #... your pytorch net definition
#        pass

# m = Net()
 
# s = b'....very long string of b64encoded compressed model goes here...' 
# di = pickle.loads(zlib.decompress(base64.b64decode(s)))
# m.load_state_dict(di)
# m.eval()
# return m

#https://www.kaggle.com/competitions/connectx/discussion/154581



# $ tar cvfz submit.tar.gz main.py data.pkl
# $ kaggle competitions submit -c connectx -f submit.tar.gz

# cwd = '/kaggle_simulations/agent/'
# if os.path.exists(cwd):
#   sys.path.append(cwd)
# else:
#   cwd = ''

# ...

# data = pickle.load(open(cwd + 'data.pkl'))


#https://www.kaggle.com/code/jamesmcguigan/pickle-zip-base64-py-file-encoding