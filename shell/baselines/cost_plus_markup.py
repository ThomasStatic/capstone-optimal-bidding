from shell.action_space import ActionSpace


class CostPlusMarkupPolicy:
    """
    price = marginal_cost + markup
    quantity = fixed quantity (MW)
    Returns a discrete action index compatible with MarketModel.
    """

    action_space: ActionSpace
    marginal_cost: float
    markup: float
    quantity_mw: float

    def act(self) -> int:
        price = float(self.marginal_cost + self.markup)
        qty = float(self.quantity_mw)
        return self.action_space.encode_from_values(price=price, quantity=qty)