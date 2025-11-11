from templating import gateway


class ExclusiveChoiceTemplate(gateway.BaseGatewayRuleTemplate):
    def gateway_type(self) -> str:
        return "Exclusive"

    def leading_clause(self):
        return "one of"

    def join_clause(self):
        return "or"


class ExplicitMergeTemplate(gateway.BaseGatewayMergeRuleTemplate):
    def gateway_type(self) -> str:
        return "Exclusive"

    def leading_clause(self):
        return "one of"

    def join_clause(self):
        return "or"


class ImplicitMergeTemplate(gateway.BaseGatewayMergeRuleTemplate):
    def gateway_type(self) -> str:
        return "Activity"

    def leading_clause(self):
        return "one of"

    def join_clause(self):
        return "or"


if __name__ == "__main__":
    def main():
        pass

    main()