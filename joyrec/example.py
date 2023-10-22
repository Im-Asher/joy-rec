from joydata.movielens import Movielens

def main():
    movielens  = Movielens(mode="train")
    for i in range(10):
        category, title, rating = movielens[i][-3:]
        print(f'category:{category}')

if __name__=="__main__":
    main()