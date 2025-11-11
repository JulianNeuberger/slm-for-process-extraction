from templating import gateway


class InclusiveSplitRuleTemplate(gateway.BaseGatewayRuleTemplate):
    def gateway_type(self) -> str:
        return "Inclusive"

    def leading_clause(self):
        return "at least one of"

    def join_clause(self):
        return "or"


class StructuredSynchronizingMergeRuleTemplate(gateway.BaseGatewayMergeRuleTemplate):
    def gateway_type(self) -> str:
        return "Inclusive"

    def leading_clause(self):
        return "at least one of"

    def join_clause(self):
        return "or"
