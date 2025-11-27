from krkn_ai.utils.rng import rng
from krkn_ai.models.scenario.base import Scenario
from krkn_ai.models.scenario.parameters import *


class SynFloodScenario(Scenario):
    name: str = "syn-flood"
    krknctl_name: str = "syn-flood"
    krknhub_image: str = "quay.io/krkn-chaos/krkn-syn-flood:latest"

    packet_size: SynFloodPacketSizeParameter = SynFloodPacketSizeParameter()
    window_size: SynFloodWindowSizeParameter = SynFloodWindowSizeParameter()
    chaos_duration: TotalChaosDurationParameter = TotalChaosDurationParameter()
    namespace: NamespaceParameter = NamespaceParameter()
    target_service: SynFloodTargetServiceParameter = SynFloodTargetServiceParameter()
    target_port: SynFloodTargetPortParameter = SynFloodTargetPortParameter()
    target_service_label: SynFloodTargetServiceLabelParameter = SynFloodTargetServiceLabelParameter()
    number_of_pods: SynFloodNumberOfPodsParameter = SynFloodNumberOfPodsParameter()
    image: SynFloodImageParameter = SynFloodImageParameter()
    node_selectors: SynFloodNodeSelectorsParameter = SynFloodNodeSelectorsParameter()

    def __init__(self, **data):
        super().__init__(**data)
        self.mutate()

    @property
    def parameters(self):
        return [
            self.packet_size,
            self.window_size,
            self.chaos_duration,
            self.namespace,
            self.target_service,
            self.target_port,
            self.target_service_label,
            self.number_of_pods,
            self.image,
            self.node_selectors,
        ]

    def mutate(self):
        namespace_candidates = [
            ns for ns in self._cluster_components.namespaces
            if getattr(ns, "services", None)
            and any(service.ports for service in ns.services)
        ]

        if namespace_candidates:
            namespace = rng.choice(namespace_candidates)
            self.namespace.value = namespace.name

            services_with_ports = [
                service for service in namespace.services if service.ports
            ]
            service = rng.choice(services_with_ports)
            self.target_service.value = service.name

            available_ports = [port.port for port in service.ports if port.port]
            self.target_port.value = rng.choice(available_ports) if available_ports else self.target_port.value
            self.target_service_label.value = ""
            return
