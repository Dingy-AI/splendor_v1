from splendor_v1.env.core.card import Card
from splendor_v1.env.core.noble import Noble
from splendor_v1.env.core.enums import GemColor

    # WHITE = 0
    # BLUE = 1
    # GREEN = 2
    # RED = 3
    # BLACK = 4
    # GOLD = 5


BASE_TIER_1 = [
    Card(
        id=0,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=1,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 3,
        }
    ),
    Card(
        id=2,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 3,
            GemColor.BLACK: 0,
        }
    ),         
    Card(
        id=3,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=4,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 3,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=5,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 2,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=6,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 2,
        }
    ),       
    Card(
        id=7,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 1,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=8,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 2,
            GemColor.GREEN: 1,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),  
    Card(
        id=9,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 2,
            GemColor.RED: 1,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=10,
        tier=1,
        points=1,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 4,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),        
    Card(
        id=11,
        tier=1,
        points=1,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 4,
            GemColor.BLACK: 0,
        }
    ),    
    Card(
        id=12,
        tier=1,
        points=1,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 4,
        }
    ),     
    Card(
        id=13,
        tier=1,
        points=1,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 4,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),     
    Card(
        id=14,
        tier=1,
        points=1,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 4,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),          
    Card(
        id=15,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 2,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 2,
        }
    ),         
    Card(
        id=16,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 2,
            GemColor.RED: 0,
            GemColor.BLACK: 2,
        }
    ),  
    Card(
        id=17,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 2,
            GemColor.GREEN: 0,
            GemColor.RED: 2,
            GemColor.BLACK: 0,
        }
    ),      
    Card(
        id=18,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 2,
            GemColor.BLACK: 0,
        }
    ),         
    Card(
        id=19,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 0,
            GemColor.GREEN: 2,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=20,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=21,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 0,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=22,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 0,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        }
    ),        
    Card(
        id=23,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 0,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=24,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 0,
        }
    ),        
    Card(
        id=25,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 2,
            GemColor.GREEN: 2,
            GemColor.RED: 0,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=26,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 0,
            GemColor.GREEN: 2,
            GemColor.RED: 2,
            GemColor.BLACK: 0,
        }
    ),     
    Card(
        id=27,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 1,
            GemColor.GREEN: 0,
            GemColor.RED: 2,
            GemColor.BLACK: 2,
        }
    ),       
    Card(
        id=28,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 0,
            GemColor.GREEN: 1,
            GemColor.RED: 0,
            GemColor.BLACK: 2,
        }
    ), 
    Card(
        id=29,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 2,
            GemColor.GREEN: 0,
            GemColor.RED: 1,
            GemColor.BLACK: 0,
        }
    ), 
    Card(
        id=30,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 1,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 1,
        }
    ),   
    Card(
        id=31,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 1,
            GemColor.GREEN: 3,
            GemColor.RED: 1,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=32,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 3,
            GemColor.GREEN: 1,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),      
    Card(
        id=33,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 1,
            GemColor.BLACK: 3,
        }
    ),  
    Card(
        id=34,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 1,
            GemColor.RED: 3,
            GemColor.BLACK: 1,
        }
    ),        
    Card(
        id=35,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 1,
            GemColor.GREEN: 2,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        }
    ),              
    Card(
        id=36,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 0,
            GemColor.GREEN: 1,
            GemColor.RED: 2,
            GemColor.BLACK: 1,
        }
    ),   
    Card(
        id=37,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 0,
            GemColor.RED: 1,
            GemColor.BLACK: 2,
        }
    ),               
    Card(
        id=38,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 0,
            GemColor.BLACK: 1,
        }
    ),           
    Card(
        id=39,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 2,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 0,
        }
    ),         
]
BASE_TIER_2 = [
    Card(
        id=40,
        tier=2,
        points=2,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 5,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=41,
        tier=2,
        points=2,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 5,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),    
    Card(
        id=42,
        tier=2,
        points=2,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 5,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),        
    Card(
        id=43,
        tier=2,
        points=2,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 5,
        }
    ),  
    Card(
        id=44,
        tier=2,
        points=2,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 5,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ), 
    Card(
        id=45,
        tier=2,
        points=3,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 6,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),    
    Card(
        id=46,
        tier=2,
        points=3,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 6,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),   
    Card(
        id=47,
        tier=2,
        points=3,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 6,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),       
    Card(
        id=48,
        tier=2,
        points=3,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 6,
            GemColor.BLACK: 0,
        }
    ),   
    Card(
        id=49,
        tier=2,
        points=3,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 6,
        }
    ), 
    Card(
        id=50,
        tier=2,
        points=1,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 3,
            GemColor.RED: 2,
            GemColor.BLACK: 2,
        }
    ),      
    Card(
        id=51,
        tier=2,
        points=1,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 2,
            GemColor.GREEN: 2,
            GemColor.RED: 3,
            GemColor.BLACK: 0,
        }
    ), 
    Card(
        id=52,
        tier=2,
        points=1,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 2,
        }
    ),  
    Card(
        id=53,
        tier=2,
        points=1,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 2,
            GemColor.BLACK: 3,
        }
    ),
    Card(
        id=54,
        tier=2,
        points=1,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 2,
            GemColor.GREEN: 2,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ), 
    Card(
        id=55,
        tier=2,
        points=2,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 1,
            GemColor.RED: 4,
            GemColor.BLACK: 2,
        }
    ), 
    Card(
        id=56,
        tier=2,
        points=2,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 1,
            GemColor.BLACK: 4,
        }
    ),     
    Card(
        id=57,
        tier=2,
        points=2,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 4,
            GemColor.BLUE: 2,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 1,
        }
    ),  
   Card(
        id=58,
        tier=2,
        points=2,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 4,
            GemColor.GREEN: 2,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),     
   Card(
        id=59,
        tier=2,
        points=2,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 1,
            GemColor.GREEN: 4,
            GemColor.RED: 2,
            GemColor.BLACK: 0,
        }
    ),   
   Card(
        id=60,
        tier=2,
        points=2,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 5,
            GemColor.BLACK: 3,
        }
    ),          
   Card(
        id=61,
        tier=2,
        points=2,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 5,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),    
   Card(
        id=62,
        tier=2,
        points=2,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 5,
            GemColor.GREEN: 3,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),   
   Card(
        id=63,
        tier=2,
        points=2,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 5,
        }
    ),  
   Card(
        id=64,
        tier=2,
        points=2,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 5,
            GemColor.RED: 3,
            GemColor.BLACK: 0,
        }
    ),  
   Card(
        id=65,
        tier=2,
        points=1,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 3,
            GemColor.BLACK: 0,
        }
    ),     
   Card(
        id=66,
        tier=2,
        points=1,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 2,
            GemColor.GREEN: 3,
            GemColor.RED: 0,
            GemColor.BLACK: 3,
        }
    ),       
   Card(
        id=67,
        tier=2,
        points=1,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 0,
            GemColor.GREEN: 2,
            GemColor.RED: 3,
            GemColor.BLACK: 0,
        }
    ),       
   Card(
        id=68,
        tier=2,
        points=1,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 2,
            GemColor.BLACK: 3,
        }
    ),     
   Card(
        id=69,
        tier=2,
        points=1,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 0,
            GemColor.GREEN: 3,
            GemColor.RED: 0,
            GemColor.BLACK: 2,
        }
    ),                                                                                                                  
]



BASE_TIER_3 = [
   Card(
        id=70,
        tier=3,
        points=4,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 7,
        }
    ),    
   Card(
        id=71,
        tier=3,
        points=4,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 7,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),    
   Card(
        id=72,
        tier=3,
        points=4,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 7,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),    
   Card(
        id=73,
        tier=3,
        points=4,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 7,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),        
   Card(
        id=74,
        tier=3,
        points=4,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 7,
            GemColor.BLACK: 0,
        }
    ),   
   Card(
        id=75,
        tier=3,
        points=5,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 7,
        }
    ),   
   Card(
        id=76,
        tier=3,
        points=5,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 7,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),       
   Card(
        id=77,
        tier=3,
        points=5,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 7,
            GemColor.GREEN: 3,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),      
   Card(
        id=78,
        tier=3,
        points=5,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 7,
            GemColor.RED: 3,
            GemColor.BLACK: 0,
        }
    ),   
   Card(
        id=79,
        tier=3,
        points=5,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 7,
            GemColor.BLACK: 3,
        }
    ),    
   Card(
        id=80,
        tier=3,
        points=4,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 3,
            GemColor.BLACK: 6,
        }
    ),        
   Card(
        id=81,
        tier=3,
        points=4,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 6,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 3,
        }
    ),    
   Card(
        id=82,
        tier=3,
        points=4,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 6,
            GemColor.GREEN: 3,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),    
   Card(
        id=83,
        tier=3,
        points=4,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 3,
            GemColor.GREEN: 6,
            GemColor.RED: 3,
            GemColor.BLACK: 0,
        }
    ),        
    Card(
        id=84,
        tier=3,
        points=4,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 3,
            GemColor.RED: 6,
            GemColor.BLACK: 3,
        }
    ),        
    Card(
        id=85,
        tier=3,
        points=3,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 3,
            GemColor.GREEN: 3,
            GemColor.RED: 5,
            GemColor.BLACK: 3,
        }
    ),  
    Card(
        id=86,
        tier=3,
        points=3,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 0,
            GemColor.GREEN: 3,
            GemColor.RED: 3,
            GemColor.BLACK: 5,
        }
    ), 
    Card(
        id=87,
        tier=3,
        points=3,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 5,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 3,
            GemColor.BLACK: 3,
        }
    ),     
    Card(
        id=88,
        tier=3,
        points=3,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 5,
            GemColor.GREEN: 3,
            GemColor.RED: 0,
            GemColor.BLACK: 3,
        }
    ),
    Card(
        id=89,
        tier=3,
        points=3,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 3,
            GemColor.GREEN: 5,
            GemColor.RED: 3,
            GemColor.BLACK: 0,
        }
    ),                                                                           
]



NOBLES = [

    Noble(
        id=0,
        Name="Anne of Brittany, Queen of France",
        points=3,
        requirement={
            GemColor.WHITE: 3,
            GemColor.BLUE: 3,
            GemColor.GREEN: 3,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Noble(
        id=1,
        Name="Catherine de' Medici, Queen of France",
        points=3,
        requirement={
            GemColor.WHITE: 3,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 3,
            GemColor.BLACK: 0,
        }
    ),
    Noble(
        id=2,
        Name="Charles V, Holy Roman Emperor",
        points=3,
        requirement={
            GemColor.WHITE: 3,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 3,
            GemColor.BLACK: 3,
        }
    ),
    Noble(
        id=3,
        Name="Elisabeth of Austria, Queen of France",
        points=3,
        requirement={
            GemColor.WHITE: 3,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 3,
        }
    ),
    Noble(
        id=4,
        Name="Francis I, King of France",
        points=3,
        requirement={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 3,
            GemColor.RED: 3,
            GemColor.BLACK: 3,
        }
    ),
    Noble(
        id=5,
        Name="Henry VIII, King of England",
        points=3,
        requirement={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 4,
            GemColor.BLACK: 4,
        }
    ),
    Noble(
        id=6,
        Name="Isabella I, Queen of Castile and Leon",
        points=3,
        requirement={
            GemColor.WHITE: 4,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 4,
        }
    ),
    Noble(
        id=7,
        Name="Mary, Queen of Scots",
        points=3,
        requirement={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 4,
            GemColor.RED: 4,
            GemColor.BLACK: 0,
        }
    ),
    Noble(
        id=8,
        Name="Niccolo Machiavelli, Diplomat",
        points=3,
        requirement={
            GemColor.WHITE: 4,
            GemColor.BLUE: 4,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Noble(
        id=9,
        Name="Suleiman the Magnificent, Sultan of the Ottoman Empire",
        points=3,
        requirement={
            GemColor.WHITE: 0,
            GemColor.BLUE: 4,
            GemColor.GREEN: 4,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
]
