from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import json
from CocktailEmbeddingMaker import Eval
from typing import List, Dict
import sys
import traceback
import random
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용, 보안을 위해 수정 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# model = tf.keras.models.load_model('testmodel.h5')
model = tf.keras.models.load_model('best_model_earthy.h5')

with open('./train_data.json', 'r') as f:
    json_data = json.load(f)

with open('./flavor.json', 'r') as f:
    flavor_data = json.load(f)
    
with open('./ingredients_description.json', 'r') as f:
    ingredients_description = json.load(f)

with open('./category.json', 'r') as f:
    category_data = json.load(f)

total_amount = 200 #ml
eval_obj = Eval(json_data, flavor_data, category_data, total_amount)

class Features(BaseModel):
    ABV: float
    sweet: float
    sour: float
    bitter: float
    spicy: float
    herbal: float
    floral: float
    fruity: float
    nutty: float
    boozy: float
    astringent: float
    umami: float
    salty: float
    perceived_t: float
    creamy: float
    smoky: float
    seed: str

class Recipe(BaseModel):
    recipe: Dict[str, float]
    profile: Dict[str, float]
    live_recipe: Dict[str, float]

class FilteredIngredients(BaseModel):
    ingredients: List[str]
    flavor: Dict[str, Dict[str, float]]
    description: Dict[str, Dict[str, str]]

class FilterRequest(BaseModel):
    features: Features
    selected_ingredient: str
    selected_index: int

def normalize_string(name):
    return name.replace('\\"', '"').replace("\\'", "'")
# 사용자의 profile을 입력으로, 가장 값이 높은 feature 3개를 뽑습니다.
# 해당 feature들과 가장 유사한 값을 가지는 ingredient를 여러개 뽑아 list로 반환합니다.
@app.post("/filter", response_model=FilteredIngredients)
async def filter(request: FilterRequest):
    try:
        features = request.features
        selected_ingredient = request.selected_ingredient
        selected_index = request.selected_index
        
        # 각 ingredient name과 이름을 제외한 ingredient flavor 정보가 맵핑되도록 합니다.
        flavor_dic = {}
        for flavor in flavor_data:
            element = {}
            for feature, value in flavor.items():
                if feature != "name" and feature != "ID":
                    element[feature] = value
            flavor_dic[flavor['name']] = element

        # 각 ingredient name과 ingredient의 카테고리 정보가 맵핑되도록 합니다.
        description_dic = {}
        for ingredient in ingredients_description:
            element = {"description": ingredient['description']}
            description_dic[ingredient['name'].lower()] = element

        user_profile = {'ABV': features.ABV,
                          'sweet' : features.sweet,
                          'sour' : features.sour,
                          'bitter' : features.bitter,
                          'spicy' : features.spicy,
                          'herbal' : features.herbal,
                          'floral' : features.floral,
                          'fruity' : features.fruity,
                          'nutty' : features.nutty,
                          'boozy' : features.boozy,
                          'astringent' : features.astringent,
                          'umami' : features.umami,
                          'salty' : features.salty,
                          'Perceived_temperature' : features.perceived_t,
                          'creamy' : features.creamy,
                          'smoky' : features.smoky,
                          }

        # 알콜이 없는 음료를 요구한 경우, ABV 정보를 제외하고 정렬된 feature 값을 생성합니다.
        sorted_values = []
        for item in user_profile.items():
            if (user_profile['ABV'] == 0 and item[0] == 'ABV'):
                continue
            sorted_values.append(item)
        sorted_values = sorted(sorted_values, key=lambda x: x[1], reverse=True)

        # 가장 특징적인 5개의 feature값을 가지는 feature들을 추출해 냅니다.
        top_5_features = sorted_values[:5]

        # 알콜이 포함되는 경우에는 ABV값도 score 계산에 포함시킵니다.
        if user_profile['ABV'] != 0: # ABV가 0이 아닌 경우에는 ABV값도 포함합니다.
            if ('ABV', user_profile['ABV']) not in top_5_features:
                top_5_features.append(('ABV', user_profile['ABV']))

        # 필터링 점수를 기준으로 정렬된 재료 목록을 생성합니다.
        # lower is better
        filter_score = {}
        for ing_name, feature_list in flavor_dic.items():
            filter_score[ing_name] = 0
            for feature, value in top_5_features:
                filter_score[ing_name] += abs(feature_list[feature] - value)

        sorted_ingredient = sorted(filter_score.items(), key=lambda x: x[1])

        # top_10_ingredient 리스트를 10개의 None 값으로 초기화합니다.
        top_10_ingredient = [None] * 10
        added_ingredients = set()

        # 사용자가 선택한 재료를 해당 인덱스에 추가하고, 추가된 재료를 기록합니다.
        if selected_ingredient and 0 <= selected_index < 10:
            top_10_ingredient[selected_index] = selected_ingredient
            added_ingredients.add(selected_ingredient)

        '''
        ABV가 0, 즉 알콜이 들어가지 않는 경우에는 재료에서 알콜이 포함되어 있는 재료를 선정해서는 안됩니다. 
        위에 top 3 feature에서 ABV값이 0이기 때문에 선택될 일이 없음으로 top 10 재료 선정에서만 주의하면 됩니다.
        '''
        # 중복 제거와 조건에 따라 top_10_ingredient 리스트를 채웁니다.
        count = 0
        # ABV가 0인 경우 (알콜이 없는 경우)
        if user_profile['ABV'] == 0:
            for ing_name, score in sorted_ingredient:
                # Mixer 카테고리에 속하고, ABV가 0인 경우에 추가합니다.
                if flavor_dic[ing_name]['ABV'] == 0 and ('Mixer' in category_data[ing_name]):
                    if ing_name in added_ingredients:
                        continue
                    while top_10_ingredient[count] is not None:
                        count += 1
                    top_10_ingredient[count] = ing_name
                    added_ingredients.add(ing_name)
                    count += 1
                if count == 10:
                    break
        else:
            # ABV가 0이 아닌 경우 (알콜이 있는 경우)
            # ABV값이 0 이상이고, Alcohol 카테고리에 속하는 재료를 선정합니다.
            seed = eval_obj.select_user_seed(user_profile)
            base_liq = ['vodka', 'tequila', 'rum']
            base = random.choice(base_liq)

            for ingredient in [base, seed]:
                if ingredient not in added_ingredients:
                    while top_10_ingredient[count] is not None:
                        count += 1
                    top_10_ingredient[count] = ingredient
                    added_ingredients.add(ingredient)
                    count += 1
                    
            for ing_name, score in sorted_ingredient:
                if flavor_dic[ing_name]['ABV'] > 0 and 'Alcohol' in category_data[normalize_string(ing_name)]:
                    if ing_name in added_ingredients:
                        continue
                    while top_10_ingredient[count] is not None:
                        count += 1
                    top_10_ingredient[count] = ing_name
                    added_ingredients.add(ing_name)
                    count += 1
                if count == 10:
                    break

        # None 값을 제거하고 최종 top 10 리스트를 생성합니다.
        top_10_ingredient = [ing for ing in top_10_ingredient if ing is not None]

        # 각 재료의 flavor와 description 정보를 생성합니다.
        top_10_flavor = {}
        top_10_description = {}
        for ing_name in top_10_ingredient:
            top_10_flavor[ing_name] = {}
            top_10_description[ing_name] = {}
            for feature, value in flavor_dic[ing_name].items():
                # 몇몇 재료는 'ID' feature가 들어가 있습니다. flavor정보에 오류가 있어 보입니다.
                if feature != "ID":
                    top_10_flavor[ing_name][feature] = value

            for feature, value in description_dic[ing_name].items():
                top_10_description[ing_name][feature] = value
                
        return {"ingredients" : top_10_ingredient,
                "flavor" : top_10_flavor,
                "description" : top_10_description}

    except Exception as e:
        _, _ , tb = sys.exc_info()    # tb  ->  traceback object
        print('file name = ', __file__)
        print('error line No = {}'.format(tb.tb_lineno))

        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=Recipe)
async def predict(features: Features):
    # Extract input features
    # Predict using the model
    try:
        input_features = {'ABV': features.ABV,
                          'sweet' : features.sweet,
                          'sour' : features.sour,
                          'bitter' : features.bitter,
                          'spicy' : features.spicy,
                          'herbal' : features.herbal,
                          'floral' : features.floral,
                          'fruity' : features.fruity,
                          'nutty' : features.nutty,
                          'boozy' : features.boozy,
                          'astringent' : features.astringent,
                          'umami' : features.umami,
                          'salty' : features.salty,
                          'Perceived_temperature' : features.perceived_t,
                          'creamy' : features.creamy,
                          'smoky' : features.smoky,
                          }
        seed_ingredient = features.seed

        print(f"eval init")
        recipe_length = 5

        generated_recipes = eval_obj.generate_recipe(model,seed_ingredient, input_features, recipe_length)
        result_recipe = {}
        print(f"generated_recipes: {generated_recipes}")

        for recipe, ingredients in zip(generated_recipes[0], generated_recipes[1]):
            result_recipe[recipe]= ingredients * total_amount

        user_recipe_profile = eval_obj.get_taste_log(generated_recipes)
        
        
        #Live Demo
        try:
            limited_ingredient_list= [  'peach schnapps','baileys irish cream','kahlua','triple sec','malibu rum','tequila',
                                'whisky','jack daniels','malibu rum','midori melon liqueur','vodka','light rum',
                                "cranberry juice","lime juice",'lemon juice',"orange juice","tonic water", "milk",'sugar syrup',
                                'powdered sugar','salt','sugar','ice','cinnamon','black pepper','grenadine','carbonated water'
                                ]
            eval_obj.set_limited_ingredient(limited_ingredient_list)
            best_ingredient = eval_obj.find_similar_ingredients(generated_recipes[0],limited_ingredient_list,input_features)
            target_abv = input_features['ABV']
            quantities = eval_obj.adjust_ingredient_quantities(best_ingredient, target_abv, input_features,total_amount=100)
            result_recipe_live = {}
            for recipe, ingredients in zip(best_ingredient, quantities):
                result_recipe_live[recipe]= ingredients * 100
        except Exception as e:
            print(traceback.format_exc())
            result_recipe_live = {}
        return {"recipe" : result_recipe,
                "profile" : user_recipe_profile,
                "live_recipe":result_recipe_live}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
