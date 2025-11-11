from templating import gateway


class ParallelSplitTemplate(gateway.BaseGatewayRuleTemplate):
    def gateway_type(self) -> str:
        return "Parallel"

    def leading_clause(self):
        return "all of"

    def join_clause(self):
        return "and"


class SynchronizationTemplate(gateway.BaseGatewayMergeRuleTemplate):
    def gateway_type(self) -> str:
        return "Parallel"

    def leading_clause(self):
        return "all of"

    def join_clause(self):
        return "and"
