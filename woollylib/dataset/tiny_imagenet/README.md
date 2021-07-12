## How to load Tiny Imagenet Dataset

### Download and Restructure folders

copy following code snipet to a `downloader.sh` file and run with command `source downloader.sh`

    #!/bin/bash

    mkdir data
    cd data

    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

    unzip tiny-imagenet-200.zip

    current="$(pwd)/tiny-imagenet-200"

    echo $current

    # training data
    cd $current/train
    for DIR in $(ls); do
        cd $DIR
        rm *.txt
        mv images/* .
        rm -r images
        cd ..
    done

    # validation data
    cd $current/val
    annotate_file="val_annotations.txt"
    length=$(cat $annotate_file | wc -l)
    for i in $(seq 1 $length); do
        # fetch i th line
        line=$(sed -n ${i}p $annotate_file)
        # get file name and directory name
        file=$(echo $line | cut -f1 -d" " )
        directory=$(echo $line | cut -f2 -d" ")
        mkdir -p $directory
        mv images/$file $directory
    done
    rm -r images
    rm val_annotations.txt
    echo "done"

This will download, extract and restructure dataset

### Get class names corresponding to folder names

For this we will first load keynames and their corresponding actual names from imagenet dataset.

    labels = {}
    with open('./data/image_net_labels.txt', 'r') as f:
        for line in f.readlines():
            labels[line.split(" ")[0]] = line.split(" ")[2][:-1].upper()

### Now we will rename folders to class names

    import os
    import glob
    for p in glob.glob('./data/tiny-imagenet-200/train/*'):
        s = p.split('/')
        os.rename(p, "/".join(s[:-1] + [labels[s[-1]]]))

    for p in glob.glob('./data/tiny-imagenet-200/val/*'):
        s = p.split('/')
        os.rename(p, "/".join(s[:-1] + [labels[s[-1]]]))


### Classes

    "ABACUS, ACADEMIC_GOWN, ACORN, AFRICAN_ELEPHANT, ALBATROSS, ALP, ALTAR, AMERICAN_ALLIGATOR, AMERICAN_LOBSTER, APRON, ARABIAN_CAMEL, BABOON, BACKPACK, BANANA, BANNISTER, BARBERSHOP, BARN, BARREL, BASKETBALL, BATHTUB, BEACH_WAGON, BEACON, BEAKER, BEE, BEER_BOTTLE, BELL_PEPPER, BIGHORN, BIKINI, BINOCULARS, BIRDHOUSE, BISON, BLACK_STORK, BLACK_WIDOW, BOA_CONSTRICTOR, BOW_TIE, BRAIN_CORAL, BRASS, BROOM, BROWN_BEAR, BUCKET, BULLET_TRAIN, BULLFROG, BUTCHER_SHOP, CANDLE, CANNON, CARDIGAN, CASH_MACHINE, CAULIFLOWER, CD_PLAYER, CENTIPEDE, CHAIN, CHEST, CHIHUAHUA, CHIMPANZEE, CHRISTMAS_STOCKING, CLIFF, CLIFF_DWELLING, COCKROACH, COMIC_BOOK, COMPUTER_KEYBOARD, CONFECTIONERY, CONVERTIBLE, CORAL_REEF, COUGAR, CRANE, DAM, DESK, DINING_TABLE, DRAGONFLY, DRUMSTICK, DUGONG, DUMBBEL, EGYPTIAN_CAT, ESPRESSO, EUROPEAN_FIRE_SALAMANDER, FLAGPOLE, FLY, FOUNTAIN, FREIGHT_CAR, FRYING_PAN, FUR_COAT, GASMASK, GAZELLE, GERMAN_SHEPHERD, GO-KART, GOLDEN_RETRIEVER, GOLDFISH, GONDOLA, GOOSE, GRASSHOPPER, GUACAMOLE, GUINEA_PIG, HOG, HOURGLASS, ICE_CREAM, ICE_LOLLY, IPOD, JELLYFISH, JINRIKISHA, KIMONO, KING_PENGUIN, KOALA, LABRADOR_RETRIEVER, LADYBUG, LAKESIDE, LAMPSHADE, LAWN_MOWER, LEMON, LESSER_PANDA, LIFEBOAT, LIMOUSINE, LION, MAGNETIC_COMPASS, MANTIS, MASHED_POTATO, MAYPOLE, MEAT_LOAF, MILITARY_UNIFORM, MINISKIRT, MONARCH, MOVING_VAN, MUSHROOM, NAIL, NECK_BRACE, OBELISK, OBOE, ORANGE, ORANGUTAN, ORGAN, OX, PARKING_METER, PAY-PHONE, PERSIAN_CAT, PICKET_FENCE, PILL_BOTTLE, PIZZA, PLATE, PLUNGER, POLE, POLICE_VAN, POMEGRANATE, PONCHO, POP_BOTTLE, POTPIE, POTTER'S_WHEEL, PRETZEL, PROJECTILE, PUNCHING_BAG, REEL, REFRIGERATOR, REMOTE_CONTROL, ROCKING_CHAIR, RUGBY_BALL, SANDAL, SCHOOL_BUS, SCOREBOARD, SCORPION, SEASHORE, SEA_CUCUMBER, SEA_SLUG, SEWING_MACHINE, SLUG, SNAIL, SNORKEL, SOCK, SOMBRERO, SPACE_HEATER, SPIDER_WEB, SPINY_LOBSTER, SPORTS_CAR, STANDARD_POODLE, STEEL_ARCH_BRIDGE, STOPWATCH, SULPHUR_BUTTERFLY, SUNGLASSES, SUSPENSION_BRIDGE, SWIMMING_TRUNKS, SYRINGE, TABBY, TAILED_FROG, TARANTULA, TEAPOT, TEDDY, THATCH, TORCH, TRACTOR, TRILOBITE, TRIUMPHAL_ARCH, TROLLEYBUS, TURNSTILE, UMBRELLA, VESTMENT, VIADUCT, VOLLEYBALL, WALKING_STICK, WATER_JUG, WATER_TOWER, WOK, WOODEN_SPOON, YORKSHIRE_TERRIER"



