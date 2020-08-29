with open (r"C:\Users\ranma\OneDrive\デスクトップ\aozora_t-master\jpn.txt", "r", encoding="utf-8")as txtdata:
    with open(r"C:\Users\ranma\OneDrive\デスクトップ\aozora_t-master\jpn-re2.txt","w",encoding="utf-8")as re:

        txt = txtdata.readlines()
        #print(txt)
        for i in txt:
            i = i.rstrip("\n")
            i = i +"\t"+"1"+"\n"
            re.write(i)
            #print(i)