# string counter test

# Write your solution here
def formatted(n):
    new = []
    for i in n:
        new.append(f"{i:.2f}")
    return new

if __name__ == "__main__":
    my_list = [1.234, 0.3333, 0.11111, 3.446]
    new_list = formatted(my_list)
    print(new_list)



print("Single string counter. Enter text to count all instances where a string occurs exactly once")
t1 = input("Enter string here: ")
leng = []
for i in t1:
    leng =+ 1
return leng
