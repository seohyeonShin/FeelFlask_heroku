import numpy as np
import pandas as pd
import random
import tensorflow as tf
import wandb
import json
import math
class CocktailEmbeddingMaker:
    def __init__(self, json_data, flavor_data,category_data, total_amount=200):
        self.cocktail_info = json_data['cocktail_info']
        self.flavor_data = flavor_data
        self.total_amount = total_amount
        self.max_recipe_length=10
        self.category_data = category_data
        self.init()
        self.ingredient_mapping=None
        self.attributes = ['ABV', 'boozy', 'sweet', 'sour', 'bitter', 'umami', 'salty', 'astringent', 'Perceived_temperature', 'spicy', 'herbal', 'floral', 'fruity', 'nutty', 'creamy', 'smoky']
    def set_ingredient_mapping(self):
        with open('limited_item_dict.json', 'r') as f:
            ingredient_mapping = json.load(f)
        self.ingredient_mapping = ingredient_mapping
    
    def normalize_string(self, name):
        return name.replace('\\"', '"').replace("\\'", "'")
        print(name)
        return name

    def init(self):
        print("CocktailEmbeddingMaker Initiated")
        ingredient_ids = {}
        try:
            for idx, item in enumerate(self.flavor_data):
                item['ID'] = idx
                normalized_name = self.normalize_string(item['name'])
                ingredient_ids[normalized_name] = idx
        except Exception as e:
            print(f"error : {e}")
        self.ingredient_ids = ingredient_ids
        self.num_ingredients = len(self.flavor_data)
        self.embedding_dim = 64
        self.low_ing = []
        self.middle_ing = []
        self.high_ing = []
        try:
            for ingredient in self.ingredient_ids.keys() :
                if self.get_ingredient_category(ingredient) == 'Alcohol':
                    ingradient_abv= self.get_ingredient_abv(ingredient)
                    #낮은 도수의 재료
                    if ingradient_abv<=15:
                        self.low_ing.append(ingredient)
                    #중간 도수의 재료
                    elif ingradient_abv>15 and ingradient_abv<30:
                        self.middle_ing.append(ingredient)
                    #높은 도수의 재료
                    else:
                        self.high_ing.append(ingredient)
        except Exception as e:
            print(f"[ingredient in self.ingredient_ids.keys()]error : {e}")
        print("CocktailEmbeddingMaker Initiated Done")

    def get_ingredient_abv(self, ingredient):
        ingredient_info = next((item for item in self.flavor_data if item["name"] == ingredient), None)
        return ingredient_info['ABV'] if ingredient_info else 0
    
    def get_ingredient_category(self,ingredient_name):
        ingredient_category = self.category_data[ingredient_name][0]
        return ingredient_category
    def create_ingredient_embedding_matrix(self):
        ingredient_embedding_matrix = np.zeros((self.num_ingredients, len(self.flavor_data[0]) - 1))
        
        for ingredient_dict in self.flavor_data:
            ingredient_name = ingredient_dict['name']
            if ingredient_name in self.ingredient_ids:
                ingredient_id = self.ingredient_ids[ingredient_name]
                ingredient_embedding = [ingredient_dict[flavor] for flavor in ingredient_dict if flavor != 'name']
                ingredient_embedding_matrix[ingredient_id] = ingredient_embedding
        
        return ingredient_embedding_matrix
    def create_recipe_embedding_1(self, recipe):
        embedding_matrix = np.random.rand(self.num_ingredients, self.embedding_dim)
        total_amount = sum(recipe.values())
        normalized_amount = {ingredient: amount / total_amount for ingredient, amount in recipe.items()}
        weighted_embeddings = []
        for ingredient, amount in normalized_amount.items():
            normalized_ingredient = self.normalize_string(ingredient)
            if normalized_ingredient not in self.ingredient_ids:
                raise KeyError(f"Ingredient '{normalized_ingredient}' not found in ingredient_ids")
            ingredient_id = self.ingredient_ids[normalized_ingredient]
            ingredient_embedding = embedding_matrix[ingredient_id]
            weighted_embedding = ingredient_embedding * amount
            weighted_embeddings.append(weighted_embedding)
        recipe_embedding = np.sum(weighted_embeddings, axis=0)
        return recipe_embedding
    def create_recipe_embedding_2(self, recipe):
        ingredient_embedding_matrix = self.create_ingredient_embedding_matrix()
        
        total_amount = sum(recipe.values())
        normalized_amount = {ingredient: amount / total_amount for ingredient, amount in recipe.items()}
        
        weighted_embeddings = []
        for ingredient, amount in normalized_amount.items():
            normalized_ingredient = self.normalize_string(ingredient)
            if normalized_ingredient not in self.ingredient_ids:
                raise KeyError(f"Ingredient '{normalized_ingredient}' not found in ingredient_ids")
            ingredient_id = self.ingredient_ids[normalized_ingredient]
            ingredient_embedding = ingredient_embedding_matrix[ingredient_id]
            weighted_embedding = ingredient_embedding * amount
            weighted_embeddings.append(weighted_embedding)
        
        recipe_embedding = np.sum(weighted_embeddings, axis=0)
        return recipe_embedding
    


    def create_recipe_embedding_list(self):
        recipe_embeddings = dict()
        for cocktail in self.cocktail_info:
            name = cocktail['cocktail_name']
            recipe = cocktail['recipe']
            recipe_embedding = self.create_recipe_embedding_2(recipe)
            recipe_embeddings[name] = {'recipe_embedding': recipe_embedding}
        return recipe_embeddings
    
    def calculate_recipe_taste_weights(self, recipe):
        recipe_ingredients = [d for d in self.flavor_data if d['name'] in list(recipe.keys())]
        total_amount = sum(recipe.values())
        # print(f"Total Amount: {total_amount} , Recipe: {recipe}")
        ingredient_ratios = {ingredient: amount / total_amount for ingredient, amount in recipe.items()}
        recipe_taste_weights = {}
        for ingredient, ratio in ingredient_ratios.items():
            ingredient_dict = next((d for d in recipe_ingredients if d['name'] == ingredient), None)
            if ingredient_dict:
                for taste, weight in ingredient_dict.items():
                    if taste != 'name':
                        recipe_taste_weights[taste] = recipe_taste_weights.get(taste, 0) + weight * ratio
        return recipe_taste_weights

    def create_taste_embedding_list(self):
        taste_embeddings = dict()
        for cocktail in self.cocktail_info:
            name = cocktail['cocktail_name']
            recipe = cocktail['recipe']
            recipe_taste_weights = self.calculate_recipe_taste_weights(recipe)
            taste_embeddings[name] = {'taste_embedding': np.array(list(recipe_taste_weights.values()))}
        return taste_embeddings
    def create_taste_embedding_pd(self):
        taste = dict()
        taste_embeddings = dict()
        for cocktail in self.cocktail_info:
            name = cocktail['cocktail_name']
            recipe = cocktail['recipe']
            recipe_taste_weights = self.calculate_recipe_taste_weights(recipe)
            taste[name] = np.array(list(recipe_taste_weights.values()))

        # 칵테일 이름과 특성 리스트 정의
        cocktail_names = list(taste_embeddings.keys())
        attributes = ['ABV', 'boozy', 'sweet', 'sour', 'bitter', 'umami', 'salty', 'astringent', 'Perceived_temperature', 'spicy', 'herbal', 'floral', 'fruity', 'nutty', 'creamy', 'smoky']
        
        # 데이터프레임 생성
        taste_embeddings = pd.DataFrame.from_dict(taste, orient='index', columns=attributes)
        return taste_embeddings
    def get_taste_info(self,cocktail_recipe):

        for ingredient in cocktail_recipe.keys():
            if ingredient not in self.ingredient_ids:
                print(f"Ingredient '{ingredient}' not found in ingredient_ids")
            else:
                recipe_taste_weights = self.calculate_recipe_taste_weights(cocktail_recipe)
                recipe_taste_weights.pop('ID')
                # print(f"[get_taste_info]recipe_taste_weights : {recipe_taste_weights}")
                return recipe_taste_weights
            
    def create_combined_embedding_list(self):
        recipe_embeddings = self.create_recipe_embedding_list()
        taste_embeddings = self.create_taste_embedding_list()

        combined_embeddings = {}
        for name in recipe_embeddings.keys():
            combined_embeddings[name] = {
                'recipe_embedding': recipe_embeddings[name]['recipe_embedding'],
                'taste_embedding': taste_embeddings[name]['taste_embedding']
            }

        return combined_embeddings
    def calculate_recipe_abv(self, recipe, quantities):
        total_amount = sum(quantities)
        total_abv = 0
        for ingredient, quantity in zip(recipe, quantities):
            ingredient_info = next((item for item in self.flavor_data if item["name"] == ingredient), None)
            if ingredient_info:
                total_abv += ingredient_info['ABV'] * (quantity / total_amount)
        return total_abv
    
    
class Eval(CocktailEmbeddingMaker):
    def __init__(self,json_data, flavor_data,category_data, total_amount=200):
        super().__init__(json_data, flavor_data,category_data, total_amount=200)
        print("Eval Class Initiated")
        self.user_seed = None
        self.user_seed_len = 0
        self.limited_ingredient_list=[]
        self.limited_mode = False

    def set_user_seed(self,user_seed_ingredient):
        self.user_seed = user_seed_ingredient
        self.user_seed_len = len(user_seed_ingredient)

    def select_user_seed(self, user_preference):
        if user_preference['ABV'] == 0:
            #Mixer중에서 선택
            user_seed = [ingredient for ingredient in self.ingredient_ids.keys() if self.get_ingredient_category(ingredient) == 'Mixer']
            if self.limited_mode:
                user_seed = [ingredient for ingredient in self.limited_ingredient_list if self.get_ingredient_category(ingredient) == 'Mixer' ]
            judge = {}                
            user_seed_list = list(set(user_seed))
            for item in user_seed_list:
                judge[item] = self.get_ingredient_taste_score(item, user_preference)
            user_seed = max(judge, key=judge.get)    
        else:
            #Alcohol중에서 선택
            if self.limited_mode:
                judge = {} 
                alcohol_list = [ingredient for ingredient in self.limited_ingredient_list if self.get_ingredient_category(ingredient) == 'Alcohol']
                # print(f"alcohol_list:{alcohol_list}")
                for item in alcohol_list:
                    judge[item] = self.get_ingredient_taste_score(item, user_preference)
                user_seed = max(judge, key=judge.get)
            else:
                user_seed_list=[]
                if user_preference['ABV']<=10:
                    user_seed_list.extend(random.choices(self.low_ing, k=3))
                    user_seed_list.extend(random.choices(self.middle_ing, k=3))
                    user_seed_list.extend(random.choices(self.high_ing, k=2))
                elif user_preference['ABV']>10 and user_preference['ABV']<=30:
                    user_seed_list.extend(random.choices(self.middle_ing, k=5))
                    user_seed_list.extend(random.choices(self.high_ing, k=2))
                else:
                    user_seed_list.extend(random.choices(self.middle_ing, k=2))
                    user_seed_list.extend(random.choices(self.high_ing, k=5))
                judge = {}                
                user_seed_list = list(set(user_seed_list))
                for item in user_seed_list:
                    judge[item] = self.get_ingredient_taste_score(item, user_preference)
                user_seed = max(judge, key=judge.get)
                # user_seed = random.choice(user_seed_list)

        print(f"user_seed : {user_seed}")
        return user_seed         
                    


    def evaluate_model(self,model, test_user_list,wandb_flag, num_recipes=100):
        self.model = model
        similarity_list= []
        diversity_list = []
        abv_match_list = []
        taste_match_list = []
        recipe_profile_list=[]
        recipe_ingredient_count_list = []
        for user in test_user_list:
            # def generate_recipe(self, seed_ingredient, user_preference, max_length=10):
            # seed_ingredient=random.choice(list(self.ingredient_ids.keys()))
            seed_ingredient = self.select_user_seed(user)
            # seed_ingredient = "vodka"#"lemon juice"
            print(seed_ingredient)
            generated_recipes = self.generate_recipe(model,seed_ingredient,user)
            print(generated_recipes)
            print(json.dumps(user,indent=4))
            recipe_profile=self.get_taste_log(generated_recipes)
            recipe_profile_list.append(recipe_profile)
            if wandb_flag:
                wandb.log({f"generated_recipe_{user['user_id']}": generated_recipes})
                ingredient_count = len(generated_recipes[0])
                recipe_ingredient_count_list.append(ingredient_count)
                wandb.log({f"ingredient_count_{user['user_id']}": ingredient_count})
                
            s = self.evaluate_similarity(generated_recipes)
            d = self.evaluate_diversity(generated_recipes)
            a = self.evaluate_abv_match(generated_recipes, user)
            t = self.evaluate_taste_match(generated_recipes, user)
            print(f"s : {s}, d : {d}, a : {a}, t : {t}")
            similarity_list.append(s)
            diversity_list.append(d)
            abv_match_list.append(a)
            taste_match_list.append(t)

        similarity = np.mean(similarity_list)
        diversity = np.mean(diversity_list)
        abv_match = np.mean(abv_match_list)
        taste_match = np.mean(taste_match_list)
        avg_ingredient_count = np.mean(recipe_ingredient_count_list)
        if wandb_flag:
            wandb.log({'avg_ingredient_count': avg_ingredient_count})
        evaluation_metrics = {
            'similarity': similarity,
            'diversity': diversity,
            'abv_match': abv_match,
            'taste_match': taste_match,
        }
        
        return evaluation_metrics,recipe_profile_list
    def cosine_similarity(self,vector1, vector2):
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        vector1_norm = math.sqrt(sum(x ** 2 for x in vector1))
        vector2_norm = math.sqrt(sum(x ** 2 for x in vector2))
        return dot_product / (vector1_norm * vector2_norm)
    def evaluate_similarity(self, generated_recipe):        
        recipe_dict = {}
        for item, quantity_ratio in zip(generated_recipe[0], generated_recipe[1]):
            recipe_dict[item] = quantity_ratio * self.total_amount
        # 벡터 생성 함수
        def create_vector(recipe, all_ingredients):
            return np.array([recipe.get(ing, 0) for ing in all_ingredients])

        # 모든 재료 목록 생성
        all_ingredients = set(recipe_dict.keys())
        for cocktail in self.cocktail_info:
            all_ingredients.update(cocktail['recipe'].keys())

        generated_vector = create_vector(recipe_dict, all_ingredients)

        max_similarity = 0
        for cocktail in self.cocktail_info:
            origin_vector = create_vector(cocktail['recipe'], all_ingredients)

            # 코사인 유사도 계산
            dot_product = np.dot(generated_vector, origin_vector)
            norm_product = np.linalg.norm(generated_vector) * np.linalg.norm(origin_vector)
            similarity = dot_product / norm_product if norm_product else 0
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def evaluate_diversity(self, generated_recipes):
        # 생성된 레시피와 원본 레시피 간의 다양성 계산
        ingredient_counts = {}
        for origin_recipe in self.cocktail_info:
            for ingredient in origin_recipe:
                if ingredient not in ingredient_counts:
                    ingredient_counts[ingredient] = 0
                ingredient_counts[ingredient] += 1
        
        for generated_recipe in generated_recipes:
            for ingredient in generated_recipe:
                if ingredient not in ingredient_counts:
                    ingredient_counts[ingredient] = 0
                ingredient_counts[ingredient] += 1
        
        total_ingredients = sum(ingredient_counts.values())
        ingredient_probs = [count / total_ingredients for count in ingredient_counts.values()]
        diversity = 1 - np.sum(np.square(ingredient_probs))
        return diversity

    def evaluate_abv_match(self, generated_recipes, user_preference):
        recipe_abv = self.calculate_recipe_abv(generated_recipes[0], generated_recipes[1])

        if user_preference['ABV'] == 0:
            return 1 if recipe_abv == 0 else 0  # 사용자가 알코올을 선호하지 않으면, 레시피도 0%여야 완벽한 일치

        abv_diff = abs(recipe_abv - user_preference['ABV'])
        normalized_abv_diff = abv_diff / (user_preference['ABV'] + 0.1)  # 0.1을 더해 분모가 0이 되는 것을 방지
        abv_match = max(0, 1 - normalized_abv_diff)  # 음수 값 방지와 0-1 범위 보장

        return abv_match

    def evaluate_taste_match(self, generated_recipe, user_preference):
        # 레시피의 맛 프로파일 일치도 계산 로직 구현
        taste_match_scores = []
        recipe = {}
        for item, quantity_ratio in zip(generated_recipe[0], generated_recipe[1]):
            recipe[item] = quantity_ratio * self.total_amount
            #재료-양 완성 
        #레시피의 맛 프로파일 생성
        recipe_taste = self.get_taste_info(recipe)
        # print("Recipe Taste Profile:", recipe_taste)
        #생성 레시피 맛 프로파일과 사용자 선호도 간의 맛 일치도 계산
        taste_differences = []
        for taste, user_score in user_preference.items():
            
            if taste!='ABV' and taste in recipe_taste:
                recipe_score = recipe_taste[taste]
                # 각 맛 특성별 점수 차이의 절대값 계산
                difference = abs(recipe_score - user_score)
                # 차이를 100으로 나누어 정규화
                normalized_difference = difference / 100
                taste_differences.append(normalized_difference)
            # 평균 일치도 계산 (1에서 정규화된 차이의 평균을 뺀 값)
        if taste_differences:
            average_match = 1 - np.mean(taste_differences)
        else:
            average_match = 0  # 레시피에 맛 특성 정보가 없을 경우

        return average_match
    
    def get_taste_log(self,generated_recipe):
        recipe = {}
        for item, quantity_ratio in zip(generated_recipe[0], generated_recipe[1]):
            recipe[item] = quantity_ratio * self.total_amount
            #재료-양 완성 
        #레시피의 맛 프로파일 생성
        recipe_taste = self.get_taste_info(recipe)
        return recipe_taste

    
    def calculate_recipe_taste_score(self, recipe, quantities, user_preference):
        recipe_taste_score = 0
        for ingredient, quantity in zip(recipe, quantities):
            ingredient_taste_score = self.get_ingredient_taste_score(ingredient, user_preference)
            recipe_taste_score += ingredient_taste_score * quantity
        
        recipe_taste_score /= len(recipe)  # 재료 개수로 나누어 평균 점수 계산
        return recipe_taste_score
    
    def set_limited_ingredient(self, ingredient_list):
        '''
        소지 재료 입력
        '''
        self.limited_ingredient_list = ingredient_list
        self.limited_mode = True
        # print(f"limited_ingredient_list : {self.limited_ingredient_list}")
        self.set_ingredient_mapping()

    def calculate_taste_similarity(self, taste_profile, ingredient_taste_profile, user_preference):
        similarity = 0
        # print(f"taste_profile : {taste_profile} , ingredient_taste_profile : {ingredient_taste_profile} , user_preference : {user_preference}")
        for taste, user_score in user_preference.items():
            if taste != 'abv_min' and taste != 'abv_max' and taste != 'user_id':
                recipe_score = taste_profile.get(taste, 0)
                ingredient_score = ingredient_taste_profile.get(taste, 0)
                similarity += abs(ingredient_score - recipe_score) * user_score / 100
        return 1 - similarity

    def get_ingredient_taste_profile(self, ingredient):
        ingredient_info = next((item for item in self.flavor_data if item["name"] == ingredient), None)
        if ingredient_info:
            taste_profile = {taste: ingredient_info[taste] for taste in self.attributes if taste != "name"}
            return taste_profile
        else:
            print(f"Ingredient '{ingredient}' not found in flavor_data")
            return None
        
    def find_similar_ingredients(self, recipe_ingredients, available_ingredients, user_preference):
        # 재료 매핑 처리
        best_set = set()
        non_mapped_recipe = []
        for recipe_ingredient in recipe_ingredients:
            if recipe_ingredient in self.ingredient_mapping:
                mapped_ingredient = self.ingredient_mapping[recipe_ingredient]
                if mapped_ingredient in available_ingredients:
                    best_set.add(mapped_ingredient)
                else:
                    non_mapped_recipe.append(recipe_ingredient)
            elif recipe_ingredient in available_ingredients:
                best_set.add(recipe_ingredient)
            else:
                non_mapped_recipe.append(recipe_ingredient)

        # print(f"recipe_ingredients: {recipe_ingredients}, best_set: {best_set}, non_mapped_recipe: {non_mapped_recipe}")

        for item in non_mapped_recipe:
            recipe_ingredient_profile = self.get_ingredient_taste_profile(item)
            best_similarity = -500
            best_replacement = None
            for available_ingredient in available_ingredients:
                if item != available_ingredient:
                    available_ingredient_profile = self.get_ingredient_taste_profile(available_ingredient)
                    taste_similarity = self.calculate_taste_similarity(recipe_ingredient_profile, available_ingredient_profile, user_preference)
                    # print(f"item: {item}, available_ingredient: {available_ingredient}, taste_similarity: {taste_similarity}")
                    if taste_similarity > best_similarity:
                        best_similarity = taste_similarity
                        best_replacement = available_ingredient
                else:
                    best_replacement = available_ingredient
                    break
            if best_replacement is None:
                # print(f"best_replacement is None for item: {item}, available_ingredients: {available_ingredients}")
                best_replacement = random.choice(available_ingredients)
            best_set.add(best_replacement)
        return best_set
    

    

    def generate_recipe(self,model, seed_ingredient, user_preference, max_length=10):
        #TODO : 가니시 고려해야함 
        #TODO : 높은 도수의 음료는 한두가지로 제한해야함
        self.model = model
        generated_recipe = [seed_ingredient]
        high_abv_count = 0
        
        max_high_abv = 3
        total_prob = 0
        max_prob_sum = 1.5
        print(f"[generate_recipe]seed_ingredient : {seed_ingredient}")
        while total_prob < max_prob_sum:
            try:
                sequence = [self.ingredient_ids[self.normalize_string(ingredient)] for ingredient in generated_recipe]
                sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=self.max_recipe_length)
            except Exception as e:
                print(f"generated_recipe : {generated_recipe}")
            print(f"sequence : {sequence}")
            probabilities = self.model.predict(sequence)[0]
            probabilities[sequence[0]] = 0  # 중복 재료 제거
            try:
                # 사용자 선호도를 반영하여 재료 선택 확률 조정
                for ingredient_id, prob in enumerate(probabilities):
                    ingredient_name = list(self.ingredient_ids.keys())[list(self.ingredient_ids.values()).index(ingredient_id)]
                    ingredient_taste_score = self.get_ingredient_taste_score(ingredient_name, user_preference)
                    ingredient_abv = self.get_ingredient_abv(ingredient_name)
                    if user_preference['ABV'] == 0 and ingredient_abv > 0:
                        abv_score = 0
                    else:
                        abv_diff = abs(ingredient_abv - user_preference['ABV'])
                        abv_score = 1 / (1 + abv_diff)  # 도수 차이가 작을수록 높은 점수
                    # 재료의 카테고리를 고려하는 후처리
                    category = self.get_ingredient_category(ingredient_name)
                    if user_preference['ABV']>0:
                        #도수가 있는 것을 선호할때
                        if category in ['Alcohol'] and ingredient_abv>32:
                            if high_abv_count >= max_high_abv:
                                probabilities[ingredient_id] *= 0.8  # 높은 도수 음료 제한
                            else:
                                high_abv_count += 1
                        elif category in ['Mixer']:
                            if high_abv_count >= max_high_abv:
                                probabilities[ingredient_id] *= 1.5
                        elif category in ['Condiment'] and total_prob>1.0:
                            probabilities[ingredient_id] *= 1.5
                    else:
                        if category in ['Alcohol']:
                            probabilities[ingredient_id] *= 0  # 높은 도수 음료 제한
                        elif category in ['Mixer']:
                            probabilities[ingredient_id] *= 2.5
                        elif category in ['Condiment'] and total_prob>1.0:
                            probabilities[ingredient_id] *= 2.5
                        
                    # elif category in ['Condiment']:
                    #     probabilities[ingredient_id] *= 10  # 과일이나 향신료, Mixer 선호
                    probabilities[ingredient_id] *= ingredient_taste_score * abv_score
            except Exception as e:
                print(f"[generate_recipe]error : {e}")
            sum_prob = sum(probabilities)
            normalized_prob = [prob / sum_prob for prob in probabilities]
            next_ingredient_id = np.argmax(normalized_prob)

            next_ingredient = list(self.ingredient_ids.keys())[list(self.ingredient_ids.values()).index(next_ingredient_id)]
            generated_recipe.append(next_ingredient)
            print(f"next_ingredient : {next_ingredient}, total_prob : {total_prob} , normalized_prob[next_ingredient_id] : {normalized_prob[next_ingredient_id]}")
            total_prob += normalized_prob[next_ingredient_id]
            if len(generated_recipe)>=max_length:
                break
        # 레시피 도수 계산 및 재료 양 조정
        target_abv = user_preference['ABV']
        quantities = self.adjust_ingredient_quantities(generated_recipe, target_abv,user_preference)
        return generated_recipe, quantities
        
        #TODO : taste고려해야함 
    def adjust_ingredient_quantities(self, recipe, target_abv, user_preference, total_amount=200, max_iterations=100):
        quantities = [total_amount / len(recipe)] * len(recipe)  # 초기 재료 양 설정 (균등 분배)
        min_quantity = 10  # 최소 재료 양 설정 (ml 단위)
        min_ingredients = 3  # 최소 재료 개수 설정
        convergence_threshold = 0.01  # 수렴 조건 설정
        max_ingredient_ratio = 1 - (len(recipe) - 1) * 0.1  # 단일 재료의 최대 비율 동적 설정

        prev_quantities = quantities.copy()  # 이전 단계의 재료 양 저장

        for iteration in range(max_iterations):
            recipe_abv = self.calculate_recipe_abv(recipe, quantities)
            recipe_taste_score = self.calculate_recipe_taste_score(recipe, quantities, user_preference)

            if abs(recipe_abv - target_abv) < convergence_threshold and recipe_taste_score >= 0.8:
                non_zero_ingredients = sum(1 for q in quantities if q > min_quantity)
                if non_zero_ingredients >= min_ingredients:
                    break
                else:
                    # 최소 재료 개수 조건을 만족하지 않는 경우, 마지막으로 유효했던 재료 양으로 되돌림
                    quantities = prev_quantities.copy()
                    break

            # 도수 차이에 따라 재료 양 조정
            abv_diff = recipe_abv - target_abv
            scale_factor = min(abs(abv_diff), 0.1) * total_amount  # 스케일링 팩터 조정

            if recipe_abv < target_abv:
                # 알코올 함량이 높은 재료의 양을 증가
                for i, ingredient in enumerate(recipe):
                    ingredient_info = next((item for item in self.flavor_data if item["name"] == ingredient), None)
                    if ingredient_info and ingredient_info['ABV'] > 0:
                        quantities[i] += scale_factor * ingredient_info['ABV'] * 0.5  # 도수의 영향을 줄임
            else:
                # 알코올 함량이 낮은 재료의 양을 증가
                for i, ingredient in enumerate(recipe):
                    ingredient_info = next((item for item in self.flavor_data if item["name"] == ingredient), None)
                    if ingredient_info and ingredient_info['ABV'] == 0:
                        quantities[i] += scale_factor * 0.5  # 도수의 영향을 줄임

            # 사용자 선호도에 따라 재료 양 조정
            for i, ingredient in enumerate(recipe):
                ingredient_taste_score = self.get_ingredient_taste_score(ingredient, user_preference)
                if ingredient_taste_score < 0.5:
                    quantities[i] = max(quantities[i] - scale_factor, min_quantity)
                elif ingredient_taste_score > 0.8:
                    quantities[i] += scale_factor * 1.5  # 사용자 선호도의 영향을 늘림

            # 최소 재료 양 조건 적용
            quantities = [max(q, min_quantity) for q in quantities]

            # 단일 재료의 비율이 최대 비율을 초과하지 않도록 조정
            max_quantity = max(quantities)
            if max_quantity / total_amount > max_ingredient_ratio:
                scale_factor = (max_ingredient_ratio * total_amount) / max_quantity
                quantities = [q * scale_factor for q in quantities]

            prev_quantities = quantities.copy()  # 현재 단계의 재료 양 저장

        # 최소 재료 개수 조건 확인
        non_zero_ingredients = sum(1 for q in quantities if q > 0)
        if non_zero_ingredients < min_ingredients:
            # 최소 재료 개수 조건을 만족하도록 재료 양 조정
            zero_indices = [i for i, q in enumerate(quantities) if q == 0]
            non_zero_indices = [i for i, q in enumerate(quantities) if q > 0]
            remaining_amount = total_amount - sum(quantities)

            while non_zero_ingredients < min_ingredients and zero_indices:
                index = zero_indices.pop(0)
                quantities[index] = min_quantity
                non_zero_ingredients += 1
                remaining_amount -= min_quantity

            # 나머지 양을 선호도에 따라 분배
            preference_scores = [self.get_ingredient_taste_score(recipe[i], user_preference) for i in non_zero_indices]
            total_preference_score = sum(preference_scores)
            for i, score in zip(non_zero_indices, preference_scores):
                quantities[i] += remaining_amount * (score / total_preference_score)

        # 총량 대비 비율로 정규화
        total_quantity = sum(quantities)
        quantities = [q / total_quantity for q in quantities]

        return quantities
    


    def get_ingredient_taste_score(self, ingredient_name, user_preference):
        ingredient_info = next((item for item in self.flavor_data if item["name"] == ingredient_name), None)

        if ingredient_info:
            # ABV 점수 계산
            abv_diff = abs(ingredient_info['ABV'] - user_preference['ABV'])
            #abv max값은 75.5
            abv_score = 1 - (abv_diff / 75.5)  # 0~1 범위로 정규화

            # 맛 점수 계산
            taste_scores = []
            for taste in user_preference:
                if taste != 'ABV' and taste != 'abv_min' and taste != 'abv_max' and taste != 'user_id':
                    ingredient_taste = ingredient_info[taste] / 100
                    user_taste = user_preference[taste] / 100
                    taste_score = 1 - abs(ingredient_taste - user_taste)  # 0~1 범위로 정규화
                    taste_scores.append(taste_score)
            
            taste_score = sum(taste_scores) / len(taste_scores)  # 맛 점수들의 평균

            # TODO: 가중치 조정 필요
            # 가중 평균 계산
            taste_weight = 0.7
            abv_weight = 0.3
            weighted_score = taste_weight * taste_score + abv_weight * abv_score

            return weighted_score
        else:
            print(f"[ingredient_name]there is no ingredient!:{ingredient_name}")
            return 0.0  # 재료 정보가 없는 경우 0 반환
    def generate_random_user_list(self,num_users):
        user_list = []
        attributes = ['ABV', 'boozy', 'sweet', 'sour', 'bitter', 'umami', 'salty', 'astringent', 'Perceived_temperature', 'spicy', 'herbal', 'floral', 'fruity', 'nutty', 'creamy', 'smoky']

        for i in range(num_users):
            user = {
                'user_id': i,
                'ABV': np.random.randint(0, 60),
            }
            for attribute in attributes[2:]:
                user[attribute] = np.random.randint(0, 100)
            user_list.append(user)

        with open(f'user_list_v1_{num_users}.json', 'w') as f:
            json.dump(user_list, f)

        print("Random user_list generated and saved.")
        return user_list
            

