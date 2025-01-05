# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 26/12/24
from config import CATEGORY_RULES, CATEGORY_MAPP_IDX_MAPP, CAR_PARTS_NAMES


class CategoryPartsRule:
    def __init__(self, name, data):
        self.name = name
        self.or_group = dict()
        self.and_group = dict()
        for group, v in data.items():
            # if group == "rear" and name == "rightRear":
            #     print(v)
            parts = set([i.strip() for i in v.split(" ")])
            if group == "mustALL":
                self.and_group[group] = parts
            else:
                self.or_group[group] = parts

    def validate_parts(self, parts, debug=False):
        parts = {CAR_PARTS_NAMES[i] for i in parts}
        if debug:
            if not isinstance(parts, set):
                parts = set(parts)
            for _, group_parts in self.and_group.items():
                if not all(part in parts for part in group_parts):
                    return False, f"and :{_}:{group_parts}"
            for _, group_parts in self.or_group.items():
                if not parts.intersection(group_parts):
                    return False, f"or :{_}:{group_parts.difference(parts)}"

            return True, None

        if not isinstance(parts, set):
            parts = set(parts)
        for _, group_parts in self.and_group.items():
            if not all(part in parts for part in group_parts):
                return False
        for _, group_parts in self.or_group.items():
            if not parts.intersection(group_parts):
                return False
        return True
    def all_parts(self):
        r = set()
        for _, group_parts in self.and_group.items():
            r = r.union(group_parts)
        for _, group_parts in self.or_group.items():
            r = r.union(group_parts)
        return r



class CategoryHandler:
    def __init__(self, data):
        self.categories = []
        for cat, v in data.items():
            self.categories.append(CategoryPartsRule(cat, v))

    def get_cat_rule(self, cat):
        for cat_rul in self.categories:
            if cat_rul.name == cat:
                return cat_rul

    def get_category(self, parts):
        for category in self.categories:
            if category.validate_parts(parts):
                return CATEGORY_MAPP_IDX_MAPP[category.name]
        return 0

CATEGORY_PARTS_RULE_HANDLER = CategoryHandler(CATEGORY_RULES)
print(CATEGORY_PARTS_RULE_HANDLER)